import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time

##############################
# CONFIGURATION
##############################
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Gesture thresholds
PINCH_THRESHOLD = 40
FIST_THRESHOLD = 120  # Distance between index tip and wrist to guess if it's a fist
SCROLL_SPEED = 30      # Speed of scrolling the canvas per movement
SCROLL_MARGIN = 50     # Movement threshold in scroll mode

# Hover dwell time (seconds) for menu selection without pinch
HOVER_SELECT_TIME = 2.0

DRAWING_SIZE = 5
ERASER_SIZE = 50

##############################
# MEDIAPIPE SETUP
##############################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

##############################
# CAMERA
##############################
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
ret, frame = cap.read()
if not ret:
    print("Error accessing camera.")
    exit()

##############################
# CANVAS SETUP
##############################
CANVAS_WIDTH = CAM_WIDTH
CANVAS_HEIGHT = 2000  # tall canvas for scrolling
current_background_name = "Blank White"

def create_lined_bg():
    lined_bg = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)*255
    for i in range(50, CANVAS_HEIGHT, 50):
        cv2.line(lined_bg, (0,i), (CANVAS_WIDTH,i), (200,200,200), 1)
    return lined_bg

# Try loading an image for background if selected
def load_image_bg():
    if os.path.exists("background.jpg"):
        img = cv2.imread("background.jpg")
        # Resize image to canvas size
        img_resized = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT))
        return img_resized
    else:
        print("background.jpg not found, using white background.")
        return np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)*255

backgrounds = {
    "Blank White": np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)*255,
    "Light Gray": np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)*230,
    "Lined Paper": create_lined_bg(),
    "Yellow Paper": np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)*255,
    "Image (if available)": "atjpg.jpg"
}

canvas = backgrounds["Blank White"].copy()

##############################
# STATE VARIABLES
##############################
prev_x, prev_y = None, None
current_color = (0,0,0)
current_size = 5
current_tool = "pen"  # 'pen' or 'eraser'

mode = "menu"  # 'menu', 'background_select', 'drawing'

scroll_mode = False
scroll_reference_y = None
canvas_offset = 0

# More colors in the toolbar:
tool_width = 60
tools = [
    {"type":"color", "color":(0,0,255), "rect":(0, 0, tool_width, 60), "label":"Red"},
    {"type":"color", "color":(0,255,0), "rect":(0, 70, tool_width, 130), "label":"Green"},
    {"type":"color", "color":(0,0,0), "rect":(0, 140, tool_width, 200), "label":"Black"},
    {"type":"color", "color":(255,0,255), "rect":(0, 210, tool_width, 270), "label":"Magenta"},
    {"type":"color", "color":(255,255,0), "rect":(0, 280, tool_width, 340), "label":"Yellow"},
    {"type":"size", "size":5, "rect":(0, 350, tool_width, 410), "label":"Size 4"},
    {"type":"size", "size":10, "rect":(0, 420, tool_width, 480), "label":"Size 8"},
    {"type":"tool", "tool":"eraser", "rect":(0, 490, tool_width, 550), "label":"Eraser"},
    {"type":"tool", "tool":"pen", "rect":(0, 560, tool_width, 620), "label":"Pen"},
    {"type":"tool", "tool":"back", "rect":(0, 630, tool_width, 690), "label":"Back"}
]

menu_buttons = [
    {"label":"Start Drawing", "rect":(CAM_WIDTH//2-100, CAM_HEIGHT//2-100, CAM_WIDTH//2+100, CAM_HEIGHT//2-40), "action":"start_drawing"},
    {"label":"Choose Background", "rect":(CAM_WIDTH//2-100, CAM_HEIGHT//2, CAM_WIDTH//2+100, CAM_HEIGHT//2+60), "action":"choose_bg"},
    {"label":"Exit", "rect":(CAM_WIDTH//2-50, CAM_HEIGHT//2+100, CAM_WIDTH//2+50, CAM_HEIGHT//2+160), "action":"exit"}
]

bg_buttons = []
y_start = 200
for i, bg_name in enumerate(backgrounds.keys()):
    y1 = y_start + i*80
    y2 = y1+60
    bg_buttons.append({
        "label":bg_name, 
        "rect":(CAM_WIDTH//2-150, y1, CAM_WIDTH//2+150, y2),
        "action":"select_bg",
        "bg_name":bg_name
    })

# For hover dwell selection in menu
hovered_button = None
hover_start_time = 0

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])*2 + (p1[1]-p2[1])*2)

def is_pinch(xi, yi, xt, yt):
    return dist((xi, yi),(xt, yt)) < PINCH_THRESHOLD

def is_fist(hand_landmarks, w, h):
    x_idx = int(hand_landmarks.landmark[8].x * w)
    y_idx = int(hand_landmarks.landmark[8].y * h)
    x_wrist = int(hand_landmarks.landmark[0].x * w)
    y_wrist = int(hand_landmarks.landmark[0].y * h)
    return dist((x_idx,y_idx),(x_wrist,y_wrist)) < FIST_THRESHOLD

def draw_toolbar(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (tool_width, CAM_HEIGHT), (200,200,200), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    for t in tools:
        x1,y1,x2,y2 = t['rect']
        if t['type'] == 'color':
            cv2.rectangle(frame,(x1,y1),(x2,y2),t['color'],-1)
        elif t['type'] == 'size':
            cv2.rectangle(frame,(x1,y1),(x2,y2),(50,50,50),-1)
            cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), t['size'], (0,0,0), -1)
        elif t['type'] == 'tool':
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,100,100),-1)
            cv2.putText(frame, t['label'], (x1+5,y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)

def draw_menu(frame):
    cv2.putText(frame, "GESTURE DRAWING MENU", (CAM_WIDTH//2-200,100), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3)
    for btn in menu_buttons:
        x1,y1,x2,y2 = btn['rect']
        cv2.rectangle(frame, (x1,y1),(x2,y2),(200,200,200),-1)
        cv2.putText(frame, btn['label'], (x1+10,y1+40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(frame, "Hover over a button for 2 sec to select", (CAM_WIDTH//2-250,CAM_HEIGHT-50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

def draw_background_menu(frame):
    cv2.putText(frame, "CHOOSE BACKGROUND", (CAM_WIDTH//2-200,100), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3)
    for btn in bg_buttons:
        x1,y1,x2,y2 = btn['rect']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(200,200,200),-1)
        cv2.putText(frame, btn['label'], (x1+10,y1+40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(frame, "Hover over a button for 2 sec to select", (CAM_WIDTH//2-300,CAM_HEIGHT-50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

def check_button_hover(x, y, buttons):
    for btn in buttons:
        x1,y1,x2,y2 = btn['rect']
        if x1<x<x2 and y1<y<y2:
            return btn
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    x_index, y_index = None, None

    if mode in ["menu","background_select"]:
        # Hover-based menu selection
        # If hovered over the same button >2s, select it automatically
        current_buttons = menu_buttons if mode=="menu" else bg_buttons
        draw_func = draw_menu if mode=="menu" else draw_background_menu
        draw_func(frame)
    elif mode == "drawing":
        draw_toolbar(frame)
        cv2.putText(frame,f"Tool: {current_tool}, Color: {current_color}, Size: {current_size}",(tool_width+10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(frame,"Open hand to draw, make fist to scroll, hover and pinch on toolbar to pick tool/color.",(tool_width+10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape
            xi = int(hand_landmarks.landmark[8].x * w)
            yi = int(hand_landmarks.landmark[8].y * h)
            xt = int(hand_landmarks.landmark[4].x * w)
            yt = int(hand_landmarks.landmark[4].y * h)

            pinch = is_pinch(xi, yi, xt, yt)
            fist = is_fist(hand_landmarks, w, h)

            if mode in ["menu","background_select"]:
                hovered = check_button_hover(xi, yi, current_buttons)
                # Highlight hovered button
                if hovered:
                    x1,y1,x2,y2 = hovered['rect']
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,255),3)
                    if hovered_button != hovered:
                        hovered_button = hovered
                        hover_start_time = time.time()
                    else:
                        # same button hovered, check time
                        if time.time() - hover_start_time > HOVER_SELECT_TIME:
                            # Select this option
                            if mode=="menu":
                                if hovered['action'] == "start_drawing":
                                    mode = "drawing"
                                elif hovered['action'] == "choose_bg":
                                    mode = "background_select"
                                elif hovered['action'] == "exit":
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    exit()
                            else:
                                # background select
                                selected_bg = hovered['bg_name']
                                if selected_bg == "Image (if available)":
                                    backgrounds["Image (if available)"] = load_image_bg()
                                    canvas = backgrounds["Image (if available)"].copy()
                                else:
                                    canvas = backgrounds[selected_bg].copy()
                                current_background_name = selected_bg
                                mode = "menu"
                            hovered_button = None
                    # Show a progress bar of hover selection
                    elapsed = time.time() - hover_start_time
                    progress = min(int((elapsed/HOVER_SELECT_TIME)*100),100)
                    cv2.rectangle(frame, (x1,y2+10), (x1+progress, y2+30), (0,255,0), -1)
                    cv2.rectangle(frame, (x1,y2+10), (x1+100, y2+30), (0,0,0), 2)
                else:
                    hovered_button = None
            elif mode=="drawing":
                hovered_tool = None
                # Check if hovering toolbar
                if xi < tool_width:
                    for t in tools:
                        x1,y1,x2,y2 = t['rect']
                        if x1<xi<x2 and y1<yi<y2:
                            hovered_tool = t
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),3)
                            if pinch:
                                # select tool/color/size
                                if t['type'] == 'color':
                                    current_color = t['color']
                                    current_tool = "pen"
                                elif t['type'] == 'size':
                                    current_size = t['size']
                                    current_tool = "pen"
                                elif t['type'] == 'tool':
                                    if t['tool'] == "eraser":
                                        current_tool = "eraser"
                                    elif t['tool'] == "pen":
                                        current_tool = "pen"
                                    elif t['tool'] == "back":
                                        mode = "menu"
                else:
                    # Drawing or scrolling
                    if fist and not scroll_mode:
                        scroll_mode = True
                        scroll_reference_y = yi
                    elif fist and scroll_mode:
                        # already in scroll mode
                        dy = yi - scroll_reference_y
                        if abs(dy) > SCROLL_MARGIN:
                            canvas_offset -= int(np.sign(dy)*SCROLL_SPEED)
                            canvas_offset = max(0, min(canvas_offset, CANVAS_HEIGHT - CAM_HEIGHT))
                            scroll_reference_y = yi
                    else:
                        if scroll_mode and not fist:
                            # exit scroll mode
                            scroll_mode = False

                        if not pinch and not fist:
                            # draw on canvas
                            cx = xi
                            cy = yi + canvas_offset
                            if current_tool == "pen":
                                if prev_x is not None and prev_y is not None:
                                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), current_color, current_size)
                                prev_x, prev_y = cx, cy
                            elif current_tool == "eraser":
                                if prev_x is not None and prev_y is not None:
                                    cv2.line(canvas,(prev_x, prev_y),(cx, cy),(255,255,255),ERASER_SIZE)
                                prev_x, prev_y = cx, cy
                        else:
                            prev_x, prev_y = None, None
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None
        hovered_button = None

    if mode == "drawing":
        visible_canvas = canvas[canvas_offset:canvas_offset+CAM_HEIGHT, 0:CAM_WIDTH]
        final_view = visible_canvas.copy()
        cv2.addWeighted(frame,0.5,final_view,0.5,0,final_view)
        frame = final_view

    cv2.imshow("Gesture Drawing App", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
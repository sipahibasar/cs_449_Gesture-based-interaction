import cv2
import mediapipe as mp
import numpy as np
import math
import time

##############################
# CONFIGURATION
##############################
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Gesture thresholds
PINCH_THRESHOLD = 40        # Distance threshold to detect a pinch
FIST_THRESHOLD = 120         # Distance threshold to detect a fist
SCROLL_SPEED = 80            # Pixels to scroll per movement
SCROLL_MARGIN = 20           # Minimum movement to trigger scrolling

# Hover dwell time (seconds) for menu and toolbar selection
HOVER_SELECT_TIME = 2.0

# Drawing parameters
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
mp_draw = mp.solutions.drawing_utils

##############################
# CAMERA SETUP
##############################
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
ret, frame = cap.read()
if not ret:
    print("Error accessing camera.")
    cap.release()
    exit()

##############################
# CANVAS SETUP
##############################
CANVAS_WIDTH = CAM_WIDTH
CANVAS_HEIGHT = 2000  # Height for scrolling
canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255  # White background

##############################
# STATE VARIABLES
##############################
prev_x, prev_y = None, None
current_color = (0, 0, 0)  # Default color: Black
current_size = DRAWING_SIZE
current_tool = "pen"       # 'pen' or 'eraser'
mode = "menu"               # Modes: 'menu' or 'drawing'

scroll_mode = False
scroll_reference_y = None
canvas_offset = 0

# Toolbar configuration
tool_width = 60
tools = [
    {"type": "color", "color": (0, 0, 255),   "rect": (0, 0, tool_width, 60),   "label": "Red"},
    {"type": "color", "color": (0, 255, 0),   "rect": (0, 70, tool_width, 130), "label": "Green"},
    {"type": "color", "color": (0, 0, 0),     "rect": (0, 140, tool_width, 200), "label": "Black"},
    {"type": "color", "color": (255, 0, 255), "rect": (0, 210, tool_width, 270), "label": "Magenta"},
    {"type": "color", "color": (255, 255, 0), "rect": (0, 280, tool_width, 340), "label": "Yellow"},
    {"type": "size",  "size": 5,             "rect": (0, 350, tool_width, 410), "label": "Size 5"},
    {"type": "size",  "size": 10,            "rect": (0, 420, tool_width, 480), "label": "Size 10"},
    {"type": "tool",  "tool": "eraser",      "rect": (0, 490, tool_width, 550), "label": "Eraser"},
    {"type": "tool",  "tool": "pen",         "rect": (0, 560, tool_width, 620), "label": "Pen"},
    {"type": "tool",  "tool": "back",        "rect": (0, 630, tool_width, 690), "label": "Back"}
]

# Menu buttons: Start Drawing and Exit
menu_buttons = [
    {"label": "Start Drawing", "rect": (CAM_WIDTH // 2 - 150, CAM_HEIGHT // 2 - 100, CAM_WIDTH // 2 + 150, CAM_HEIGHT // 2 - 40), "action": "start_drawing"},
    {"label": "Exit", "rect": (CAM_WIDTH // 2 - 100, CAM_HEIGHT // 2 + 100, CAM_WIDTH // 2 + 100, CAM_HEIGHT // 2 + 160), "action": "exit"}
]

# For hover dwell selection in menu and toolbar
hovered_button = None
hover_start_time = 0

##############################
# HELPER FUNCTIONS
##############################
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_pinch(xi, yi, xt, yt):
    """Determine if a pinch gesture is made based on fingertip distances."""
    return calculate_distance((xi, yi), (xt, yt)) < PINCH_THRESHOLD

def is_fist(hand_landmarks, frame_width, frame_height):
    """Determine if the hand gesture is a fist based on landmark distances."""
    x_index = int(hand_landmarks.landmark[8].x * frame_width)
    y_index = int(hand_landmarks.landmark[8].y * frame_height)
    x_wrist = int(hand_landmarks.landmark[0].x * frame_width)
    y_wrist = int(hand_landmarks.landmark[0].y * frame_height)
    return calculate_distance((x_index, y_index), (x_wrist, y_wrist)) < FIST_THRESHOLD

def draw_toolbar_ui(frame):
    """Draw the toolbar with tools, colors, and sizes."""
    # Semi-transparent overlay for toolbar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tool_width, CAM_HEIGHT), (200, 200, 200), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    for tool in tools:
        x1, y1, x2, y2 = tool['rect']
        if tool['type'] == 'color':
            cv2.rectangle(frame, (x1, y1), (x2, y2), tool['color'], -1)
        elif tool['type'] == 'size':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), tool['size'], (0, 0, 0), -1)
        elif tool['type'] == 'tool':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), -1)
            cv2.putText(frame, tool['label'], (x1 + 5, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_menu_ui(frame):
    """Draw the main menu with Start Drawing and Exit buttons."""
    cv2.putText(frame, "GESTURE DRAWING MENU", (CAM_WIDTH // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    for btn in menu_buttons:
        x1, y1, x2, y2 = btn['rect']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
        cv2.putText(frame, btn['label'], (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "Hover over a button for 2 sec to select", (CAM_WIDTH // 2 - 250, CAM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def check_button_hover(x, y, buttons):
    """Check if the current cursor is hovering over any button."""
    for btn in buttons:
        x1, y1, x2, y2 = btn['rect']
        if x1 < x < x2 and y1 < y < y2:
            return btn
    return None

def handle_hover_selection(current_mode, hovered, xi, yi):
    """Handle hover-based selection for menu and toolbar."""
    global hovered_button, hover_start_time, mode, current_color, current_size, current_tool, scroll_mode
    
    if hovered != hovered_button:
        hovered_button = hovered
        hover_start_time = time.time()
    else:
        if hovered_button:
            elapsed = time.time() - hover_start_time
            if elapsed > HOVER_SELECT_TIME:
                # Perform the button's action
                if current_mode == "menu":
                    action = hovered_button['action']
                    if action == "start_drawing":
                        mode = "drawing"
                    elif action == "exit":
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                elif current_mode == "drawing":
                    # Determine the type of tool selected
                    if hovered_button['type'] == 'color':
                        current_color = hovered_button['color']
                        current_tool = "pen"
                    elif hovered_button['type'] == 'size':
                        current_size = hovered_button['size']
                        current_tool = "pen"
                    elif hovered_button['type'] == 'tool':
                        tool = hovered_button['tool']
                        if tool == "eraser":
                            current_tool = "eraser"
                        elif tool == "pen":
                            current_tool = "pen"
                        elif tool == "back":
                            mode = "menu"
                hovered_button = None
    return hovered_button is not None and (time.time() - hover_start_time) > HOVER_SELECT_TIME

def draw_cursor(frame, xi, yi):
    """Draw a visible cursor on the fingertip."""
    cv2.circle(frame, (xi, yi), 10, (0, 255, 0), -1)

##############################
# MAIN APPLICATION LOOP
##############################
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)  # Mirror the image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if mode == "menu":
        draw_menu_ui(frame)
        current_buttons = menu_buttons
    elif mode == "drawing":
        draw_toolbar_ui(frame)
        # Display current tool, color, and size
        cv2.putText(frame, f"Tool: {current_tool.capitalize()}, Color: {current_color}, Size: {current_size}", 
                    (tool_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Instructions
        cv2.putText(frame, "Hover 2s to select tool/color; Open hand to draw; Fist to scroll; Hover 'Back' to return menu", 
                    (tool_width + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            frame_height, frame_width, _ = frame.shape
            # Extract landmark coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            xi = int(index_finger_tip.x * frame_width)
            yi = int(index_finger_tip.y * frame_height)
            xt = int(thumb_tip.x * frame_width)
            yt = int(thumb_tip.y * frame_height)
            
            # Determine gestures
            pinch = is_pinch(xi, yi, xt, yt)
            fist = is_fist(hand_landmarks, frame_width, frame_height)
            
            if not scroll_mode:
                # Draw cursor only when not in scroll mode
                draw_cursor(frame, xi, yi)
            
            if mode == "menu":
                hovered = check_button_hover(xi, yi, current_buttons)
                if hovered:
                    x1, y1, x2, y2 = hovered['rect']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Highlight
                    handle_hover_selection("menu", hovered, xi, yi)
                    # Draw progress bar
                    if hovered_button:
                        elapsed = time.time() - hover_start_time
                        progress = min(int((elapsed / HOVER_SELECT_TIME) * 100), 100)
                        cv2.rectangle(frame, (x1, y2 + 10), (x1 + progress, y2 + 30), (0, 255, 0), -1)
                        cv2.rectangle(frame, (x1, y2 + 10), (x1 + 100, y2 + 30), (0, 0, 0), 2)
                else:
                    hovered_button = None
    
            elif mode == "drawing":
                if fist and not scroll_mode:
                    scroll_mode = True
                    scroll_reference_y = yi
                    # Reset previous positions to stop drawing
                    prev_x, prev_y = None, None
                elif fist and scroll_mode:
                    dy = yi - scroll_reference_y
                    if abs(dy) > SCROLL_MARGIN:
                        canvas_offset -= int(math.copysign(SCROLL_SPEED, dy))
                        canvas_offset = max(0, min(canvas_offset, CANVAS_HEIGHT - CAM_HEIGHT))
                        scroll_reference_y = yi
                else:
                    if scroll_mode and not fist:
                        # Exit scroll mode
                        scroll_mode = False
                    if not scroll_mode:
                        # Hovering over toolbar
                        if xi < tool_width:
                            hovered_tool = check_button_hover(xi, yi, tools)
                            if hovered_tool:
                                x1, y1, x2, y2 = hovered_tool['rect']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Highlight
                                handle_hover_selection("drawing", hovered_tool, xi, yi)
                                # Draw progress bar
                                if hovered_button:
                                    elapsed = time.time() - hover_start_time
                                    progress = min(int((elapsed / HOVER_SELECT_TIME) * 100), 100)
                                    cv2.rectangle(frame, (x1, y2 + 10), (x1 + progress, y2 + 30), (0, 255, 0), -1)
                                    cv2.rectangle(frame, (x1, y2 + 10), (x1 + 100, y2 + 30), (0, 0, 0), 2)
                        else:
                            # Not hovering over toolbar; handle drawing
                            if not pinch:
                                # Drawing with open hand
                                cx = xi
                                cy = yi + canvas_offset
                                if current_tool == "pen":
                                    if prev_x is not None and prev_y is not None:
                                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), current_color, current_size)
                                    prev_x, prev_y = cx, cy
                                elif current_tool == "eraser":
                                    if prev_x is not None and prev_y is not None:
                                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 255, 255), ERASER_SIZE)
                                    prev_x, prev_y = cx, cy
                            else:
                                # Pinching detected; stop drawing
                                prev_x, prev_y = None, None
    else:
        # No hands detected
        prev_x, prev_y = None, None
        hovered_button = None

    if mode == "drawing":
        # Display the portion of the canvas based on the current offset
        visible_canvas = canvas[canvas_offset:canvas_offset + CAM_HEIGHT, 0:CAM_WIDTH]
        final_view = visible_canvas.copy()
        # Overlay toolbar and instructions onto the final view
        cv2.addWeighted(frame, 0.5, final_view, 0.5, 0, final_view)
        frame = final_view

    cv2.imshow("Gesture Drawing App", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Canvas for drawing
ret, frame = cap.read()
if not ret:
    print("Camera not accessible.")
    exit()

height, width, _ = frame.shape
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

# Define Toolbar Buttons
# Let's define a simple toolbar with colors on the left side
# Each button: (x1,y1,x2,y2), color or action
# Positions assume a small width of 50px for tools.
tool_width = 50
tools = [
    {'rect':(0, 0, tool_width, 50), 'color':(0,0,255), 'type':'color'},     # Red
    {'rect':(0, 60, tool_width, 110), 'color':(0,255,0), 'type':'color'},   # Green
    {'rect':(0, 120, tool_width, 170), 'color':(255,0,0), 'type':'color'},  # Blue
    {'rect':(0, 180, tool_width, 230), 'color':(0,0,0), 'type':'color'},    # Black
    {'rect':(0, 240, tool_width, 290), 'size':2, 'type':'size'},            # Small brush
    {'rect':(0, 300, tool_width, 350), 'size':5, 'type':'size'},            # Medium brush
    {'rect':(0, 360, tool_width, 410), 'size':10, 'type':'size'}            # Large brush
]

current_color = (0,0,0) # default black
current_size = 5

prev_x, prev_y = None, None

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw the toolbar on the left side
    for tool in tools:
        x1, y1, x2, y2 = tool['rect']
        if tool['type'] == 'color':
            cv2.rectangle(frame, (x1,y1), (x2,y2), tool['color'], -1)
        else: # size
            # Represent size tool by a circle or line
            cv2.rectangle(frame, (x1,y1), (x2,y2), (50,50,50), -1)
            cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), tool['size'], (0,0,0), -1)

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract coordinates
            h, w, c = frame.shape
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)

            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            # Check pinch
            pinch_dist = dist((x_index, y_index), (x_thumb, y_thumb))
            is_pinch = pinch_dist < 40  # Threshold depends on camera distance

            # Check if hovering over a tool
            hovered_tool = None
            for tool in tools:
                x1, y1, x2, y2 = tool['rect']
                if x_index > x1 and x_index < x2 and y_index > y1 and y_index < y2:
                    hovered_tool = tool
                    # Highlight tool
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
                    break

            # If pinch and hovering over a tool, select its property
            if is_pinch and hovered_tool:
                if hovered_tool['type'] == 'color':
                    current_color = hovered_tool['color']
                elif hovered_tool['type'] == 'size':
                    current_size = hovered_tool['size']

            # Drawing on canvas
            # If not pinching and not on toolbar area, draw.
            if not is_pinch and (x_index > tool_width):  
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), current_color, current_size)
                prev_x, prev_y = x_index, y_index
            else:
                prev_x, prev_y = None, None

            # Draw current brush indicator at fingertip
            cv2.circle(frame, (x_index, y_index), current_size, current_color, -1)
    else:
        prev_x, prev_y = None, None

    # Combine canvas and frame
    # You can overlay the canvas onto the frame or show side by side.
    # Let's overlay the canvas onto the frame:
    overlay = frame.copy()
    cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0, overlay)

    cv2.putText(overlay, f"Color: {current_color}", (60,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(overlay, f"Size: {current_size}", (60,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow("Gesture Drawing App", overlay)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
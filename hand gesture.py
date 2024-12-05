import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands and Drawing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture from the computer's camera
cap = cv2.VideoCapture(0)  # Use 0 for the computer's built-in webcam

# Variables to store the previous gesture and action time
previous_action = None
last_action_time = time.time()

# Function to determine the number of fingers open
def fingers_open(hand_landmarks):
    open_fingers = 0
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Tip above knuckle
            open_fingers += 1
    return open_fingers

# Function to detect specific gestures
def detect_gesture(hand_landmarks):
    open_fingers = fingers_open(hand_landmarks)

    # Detect Flat Hand (Jump)
    if open_fingers == 4:
        return "jump"

    # Detect Fist (Duck)
    elif open_fingers == 0:
        return "duck"

    # Detect Two-Finger Gesture (Move Left)
    elif open_fingers == 2 and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        return "left"

    # Detect Three-Finger Gesture (Move Right)
    elif open_fingers == 3 and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        return "right"

    # Default to "nothing"
    return "nothing"

# Gesture-to-Action Mapping
def perform_action(gesture):
    if gesture == "jump":
        pyautogui.press('up')  # Jump in the game
    elif gesture == "duck":
        pyautogui.press('down')  # Duck in the game
    elif gesture == "left":
        pyautogui.press('left')  # Move left in the game
    elif gesture == "right":
        pyautogui.press('right')  # Move right in the game
    elif gesture == "nothing":
        print("Relaxed - No action")  # No action

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to access the camera. Check permissions!")
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    gesture = "nothing"  # Default gesture when no action is detected

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            gesture = detect_gesture(hand_landmarks)

    # Perform the action only if the gesture changes or after a time threshold
    if gesture != previous_action or time.time() - last_action_time > 0.5:
        perform_action(gesture)
        previous_action = gesture
        last_action_time = time.time()

    # Display the frame with detected gestures
    cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Hand Gesture Controlled Game", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

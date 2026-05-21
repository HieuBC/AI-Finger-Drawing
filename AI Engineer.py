import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open primary webcam (index 0)
cap = cv2.VideoCapture(0)

# Create a black canvas for drawing (same size as webcam frame)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables to store previous position and drawing mode
prev_x, prev_y = 0, 0
drawing_mode = False  # Flag to toggle drawing mode
fingers_touching = False  # Flag to check if fingers are currently touching

# Button coordinates and appearance
button_x1, button_y1 = 10, 10   # Top-left corner of the "Clear" button
button_x2, button_y2 = 150, 60  # Bottom-right corner of the "Clear" button
button_color = (0, 255, 0)      # Button color (green)

# Helper function to calculate the Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Helper function to check if the index finger is inside the button
def is_inside_button(x, y, x1, y1, x2, y2):
    return x1 < x < x2 and y1 < y < y2

# Helper function to check if both index finger and thumb are open
def fingers_are_open(landmarks):
    # Check if the tips of the index finger and thumb are above their respective knuckles
    index_tip = landmarks[8]
    index_knuckle = landmarks[6]
    thumb_tip = landmarks[4]
    thumb_knuckle = landmarks[2]

    # Both fingers are open if tips are above their knuckles in the y-axis
    return index_tip.y < index_knuckle.y and thumb_tip.y < thumb_knuckle.y

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror-like interaction
    frame = cv2.flip(frame, 1)

    # Convert frame from BGR to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Draw the "Clear" button
    cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
    # Calculate text size and center the "Clear" text inside the button
    text = "Clear"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1, 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = button_x1 + (button_x2 - button_x1 - text_size[0]) // 2
    text_y = button_y1 + (button_y2 - button_y1 + text_size[1]) // 2
    # Draw "Clear" label
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Process hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the index finger tip (landmark 8) and thumb tip (landmark 4)
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert normalized landmark coordinates to pixel coordinates
            h, w, _ = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate the distance between index finger and thumb
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)

            # Threshold for touch detection (adjustable)
            threshold = 40

            # Check if fingers are touching and both are open
            if distance < threshold:
                if not fingers_touching and fingers_are_open(hand_landmarks.landmark):
                    drawing_mode = not drawing_mode  # Toggle drawing mode
                    print(f"Drawing mode {'ON' if drawing_mode else 'OFF'}")
                    if drawing_mode:
                        time.sleep(0.2)  # Short delay after toggling drawing mode
                fingers_touching = True
            else:
                fingers_touching = False

            # If drawing mode is active, draw on the canvas
            if drawing_mode:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y
                # Draw red line between previous and current points
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 5)
                prev_x, prev_y = index_x, index_y
            else:
                # Reset previous coordinates when not drawing
                prev_x, prev_y = 0, 0

            # Clear canvas if index finger is inside the "Clear" button
            if is_inside_button(index_x, index_y, button_x1, button_y1, button_x2, button_y2):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                print("Canvas cleared!")

            # Draw hand landmarks on the frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine the frame and canvas for display
    combined_frame = cv2.add(frame, canvas)

    # Display the combined frame
    cv2.imshow("Finger Drawing", combined_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

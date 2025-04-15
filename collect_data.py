import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from collections import defaultdict

# Output file
OUTPUT_FILE = "keypoints_dataset.csv"
HEADER_CREATED = os.path.exists(OUTPUT_FILE)

# Count existing samples
sample_count = defaultdict(int)

if HEADER_CREATED:
    with open(OUTPUT_FILE, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:  # skip empty rows
                label = row[-1]
                sample_count[label] += 1

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Create header row
if not HEADER_CREATED:
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f'x{i+1}', f'y{i+1}']
        header += ['label']
        writer.writerow(header)

print("Press letter keys (A-Z) to label and save current hand pose.")
print("Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Wait for key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                label = chr(key).upper()
                sample_count[label] += 1
                print(f"Saved: {label} (Total: {sample_count[label]})")
                with open(OUTPUT_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks + [label])

    # Display class sample counts at the top-left corner
    y_offset = 30
    for label, count in sorted(sample_count.items()):
        text = f"{label}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25

    cv2.imshow("Label Hand Signs - Press A-Z", frame)

cap.release()
cv2.destroyAllWindows()

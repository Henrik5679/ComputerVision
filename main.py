import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model and label encoder
model = tf.keras.models.load_model("keypoint_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Predict only if we have 21 landmarks (42 values)
            if len(landmarks) == 42:
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                confidence = np.max(prediction)

                if confidence > 0.8:
                    prediction_text = f"{predicted_label} ({confidence:.2f})"
                else:
                    prediction_text = "Not confident"

    # Display prediction
    cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Real-Time Sign Prediction", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

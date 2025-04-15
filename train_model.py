import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("keypoints_dataset.csv")

# Split features and labels
X = df.iloc[:, :-1].values  # 42 features (x1, y1, ..., x21, y21)
y = df.iloc[:, -1].values   # last column is label

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# One-hot encode target
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Define a simple MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model and label encoder
model.save("keypoint_model.keras")

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved!")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
train_dir = "train"
test_dir = "test"

# Image dimensions
img_width, img_height = 48, 48

# Preprocess the images using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions in the dataset

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    train_data,
    epochs=25  ,
    validation_data=test_data
)
model.save("emotion_model.h5")
# Load the model
model = load_model("emotion_model.h5")

# Emotion Labels (order must match your training generator's class_indices)
emotion_labels = list(train_data.class_indices.keys())

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = grayscale_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype("float") / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        prediction = model.predict(roi_reshaped)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw Rectangle and Emotion Label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

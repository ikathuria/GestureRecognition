"""The main CNN."""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Step 1: Converting dataset
loaded_images = []

dataset_path = "data1"

for folder in os.listdir(dataset_path):
    gesture_path = os.path.join(dataset_path, folder)

    k = 0
    for img in os.listdir(gesture_path):
        image = cv2.imread(os.path.join(gesture_path, img))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (100, 100))
        loaded_images.append(gray_image)
        k += 1

print("Total images in dataset:", len(loaded_images))  # 1800

outputVectors = []

for i in range(18):
    for _ in range(0, k):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[i] = 1
        outputVectors.append(temp)

print("Output vector length:", len(outputVectors))  # 1800


X = np.asarray(loaded_images)
y = np.asarray(outputVectors)
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
print("Number of training images:", X_train.shape)
print("Number of test images:", X_test.shape)

# Step 2: Model
model = Sequential()

# first conv layer
# input shape = (img_rows, img_cols, 1)
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# second conv layer
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# third conv layer
model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

# flatten and put a fully connected layer
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.20))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.30))

# softmax layer
model.add(Dense(18, activation="softmax"))

# model summary
optimiser = Adam()
model.compile(
    optimizer=optimiser,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
model.summary()

# Step 3: Fit the model
model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=1000,
    verbose=1,
    validation_data=(X_test, y_test),
)

# Step 4: Save the model
model.save("model.h5")

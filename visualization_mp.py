"""Visualizing the results with OpenCV."""

import os
import cv2
import numpy as np

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("setting up, please wait...")

from keras.models import load_model
from image_processing import detectHands, getHands


def load_weights():
    """Load Model Weights.

    Returns:
        the loaded model if available, otherwise None.
    """
    try:
        model = load_model(latest_model)
        return model

    except Exception as e:
        return None


def get_predicted_class(model):
    """Get the predicted class.

    Args:
        model: the loaded model.

    Returns:
        the predicted class.
    """
    image = cv2.imread("temp_mp.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (320, 240))
    gray_image = gray_image.reshape(1, 240, 320, 1)

    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class].upper()


# path
latest_model = "model/" + "01May_model10.h5"

# labels in order of training output
labels = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
          5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
          10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
          15: "on", 16: "ok"}

print('switching on camera...')
cap = cv2.VideoCapture(0)

model = load_weights()

while True:
    ret, frame = cap.read()

    # removing mirror image
    frame = cv2.flip(frame, 1)
    height, width, channels = frame.shape

    # blank image
    blank_image = np.zeros((height, width, channels), np.uint8)

    # detect mediapipe hands
    mediapipe_hands = detectHands(frame, blank_image)
    if type(mediapipe_hands) != type(None):
        cv2.imwrite("temp_mp.png", mediapipe_hands)

        predicted_class = get_predicted_class(model)
        cv2.putText(frame, str(predicted_class), (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Mediapipe Hands", mediapipe_hands)
    else:
        cv2.putText(frame, "NO HANDS", (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    # on keypress
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
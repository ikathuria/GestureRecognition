"""Visualizing the results with OpenCV."""

import os
import cv2
import numpy as np

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("setting up, please wait...")

from keras.models import load_model
from image_processing import detect_hands, find_biggest_contour


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
    img = cv2.imread("temp.png")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)

    biggest = find_biggest_contour(imgThreshold)

    biggest = find_biggest_contour(imgThreshold)

    cv2.drawContours(img, biggest, -1, (255, 255, 255), 25)
    
    try:
        img = cv2.resize(img, (100, 100))
        img = img.reshape(1, 100, 100, 1)

        prediction = model.predict_on_batch(img)
        predicted_class = np.argmax(prediction)

        return labels[predicted_class].upper()
    except:
        print("Size wrong")


# path
latest_model = "model/" + "09Apr_model_24.h5"

# labels in order of training output
labels = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
          5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
          10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
          15: "on", 16: "ok"}
# labels = {0: "blank", 1: "down", 2: "eight", 3: "five", 4: "four",
#           5: "left", 6: "nine", 7: "off", 8: "ok", 9: "on", 10: "one",
#           11: "right", 12: "seven", 13: "six", 14: "three", 15: "two",
#           16: "up", 17: "zero"}

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
    mediapipe_hands = detect_hands(frame, blank_image)
    if type(mediapipe_hands) != type(None):
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

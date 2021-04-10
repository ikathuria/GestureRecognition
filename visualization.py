"""Visualizing the results with OpenCV."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("setting up, please wait...")

import cv2
import numpy as np
from keras.models import load_model
from image_processing import run_avg, segment

# accumulated weight
accumWeight = 0.5

# path
latest_model = "model/" + "10Apr_model_12.h5"

# labels in order of training output
labels = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
          5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
          10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
          15: "on", 16: "ok", 17: "blank"}
# labels = {0: "blank", 1: "down", 2: "eight", 3: "five", 4: "four",
#           5: "left", 6: "nine", 7: "off", 8: "ok", 9: "on", 10: "one",
#           11: "right", 12: "seven", 13: "six", 14: "three", 15: "two",
#           16: "up", 17: "zero"}


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


def getPredictedClass(model):
    """Get the predicted class.
    
    Args:
        model: the loaded model.
    
    Returns:
        the predicted class.
    """
    image = cv2.imread("Temp.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 100))
    gray_image = gray_image.reshape(1, 100, 100, 1)

    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class].upper()


cap = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 10, 310, 310, 610

num_frames = 0

model = load_weights()

while True:
    ret, frame = cap.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is reached
    # so that our weighted average model gets calibrated
    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("\n[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successfull...")
            print("Press 'c' to recalibrate background")
    else:
        # segment the hand region
        hand = segment(gray)

        if hand is not None:
            (thresholded, segmented) = hand
            
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

            cv2.imwrite('Temp.png', thresholded)

            predictedClass = getPredictedClass(model)

            cv2.putText(clone, str(predictedClass), (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Thesholded", thresholded)

        else:
            cv2.putText(clone, "BLANK", (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", clone)

    num_frames += 1

    # on keypress
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break
    elif keypress == ord("c"):
        num_frames = 0

cap.release()
cv2.destroyAllWindows()

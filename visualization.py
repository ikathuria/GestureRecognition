"""Visualizing the results with OpenCV."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mediapipe as mp

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("setting up, please wait...")

from keras.models import load_model
from image_processing import run_avg, segment
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# sys.path.insert(0, './AdaBins')
# from infer import InferenceHelper
# inferHelper = InferenceHelper(device='cpu')

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


def get_hands(image, x, y):
    minx = min(x)
    miny = min(y)
    maxx = max(x)
    maxy = max(y)
    
    final_coords = (minx-25, miny-25, maxx+25, maxy+25)
    # cv2.imshow("ROI", image[100:100, 100:100])

    return final_coords


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
    image = cv2.imread("temp_threshold.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 100))
    gray_image = gray_image.reshape(1, 100, 100, 1)

    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class].uper()


print('switching on camera...')
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.4,
                       min_tracking_confidence=0.4)
cap = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
# top, right, bottom, left = 10, 310, 310, 610

num_frames = 0

model = load_weights()

while True:
    ret, frame = cap.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    height, width, channels = clone.shape
    
    # blank image
    blank_image = np.zeros((height, width, channels), np.uint8)

    clone.flags.writeable = False

    results = hands.process(clone)
    clone.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            coords_x = []
            coords_y = []
            for l in landmarks:
                coords_x.append(int(l.x*width))
                coords_y.append(int(l.y*height))
        bounded_hands = get_hands(clone, coords_x, coords_y)
        
        cv2.imshow("Mediapipe hands", blank_image)
        
        # get the ROI
        # right, left = bounded_hands[0][2], bounded_hands[0][0]
        # top, bottom = bounded_hands[0][1], bounded_hands[0][3]
        # roi = clone[top:bottom, right:left]
        
        # print(roi)
        # print(type(roi))

        # # convert the roi to grayscale and blur it
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # hull = []
        # for i in range(len(contours)):
        #     hull.append(cv2.convexHull(contours[i], False))
        
        # for i in range(len(contours)):
        #     color_contours = (0, 255, 0) # green - color for contours
        #     color = (255, 0, 0) # blue - color for convex hull
        #     # draw ith contour
        #     cv2.drawContours(gray, contours, i, color_contours, 1, 8, hierarchy)
        #     # draw ith convex hull object
        #     cv2.drawContours(gray, hull, i, color, 1, 8)
        
        # if type(roi) == type(np.array):
        #     cv2.imwrite('temp_threshold.png', roi)

        # cv2.rectangle(clone, (left, top), (right, bottom), (0, 0, 0), 2)

        # predictedClass = getPredictedClass(model)

        # cv2.putText(clone, str(predictedClass), (70, 45),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.imshow("Thesholded", bounded_hands[1])
    else:
        cv2.putText(clone, "BLANK", (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", clone)

    num_frames += 1

    # on keypress
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()

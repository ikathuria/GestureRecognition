"""Functions required for real-time image processing."""

import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4
)

# global variables
bg = None


def run_avg(image, aWeight):
    """Set real-time background.

    Args:
        image: the background image.
        aWeight: accumulated weight.
    """
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    """Segment the image.

    Args:
        image: the image to be segmented.
        threshold: the threshold value, 25 by default.
    """
    global bg

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def detectHands(image, draw_image):
    """Return hand part of image.
    
    Args:
        image: the image to be processed.
        draw_image: image to display mediapipe skeleton.
    
    Returns:
        roi: image of hand.
    """
    height, width, channels = image.shape

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                draw_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            landmarks = hand_landmarks.landmark
            coords_x = []
            coords_y = []
            for l in landmarks:
                coords_x.append(int(l.x * width))
                coords_y.append(int(l.y * height))
        # bounded_hands = getHands(image, coords_x, coords_y)
        return draw_image
    return None


def getHands(image, x, y):
    """Return hand part of image.
    
    Args:
        image: the image to be processed.
        x: x coordinates.
        y: y coordinates.
    
    Returns:
        roi: image of hand.
    """
    minx = min(x)
    miny = min(y)
    maxx = max(x)
    maxy = max(y)

    final_coords = (minx - 25, miny - 25, maxx + 25, maxy + 25)

    right, left = final_coords[2], final_coords[0]
    top, bottom = final_coords[1], final_coords[3]
    roi = image[top:bottom, right:left]

    cv2.imshow("temp_fgbgs.png", roi)

import numpy as np
import cv2
import os

# global variables
bg1 = None
bg2 = None
accumWeight = 0.5

path1 = "data1/"
path2 = "data2/"


def run_avg1(image, aWeight):
    global bg1
    # initialize the background
    if bg1 is None:
        bg1 = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg1, aWeight)


def run_avg2(image, aWeight):
    global bg2
    # initialize the background
    if bg2 is None:
        bg2 = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg2, aWeight)


def segment(image, type_of_bg, threshold=25):
    global bg1, bg2

    # find the absolute difference between background and current frame    
    if type_of_bg == 1:
        diff = cv2.absdiff(bg1.astype("uint8"), image)
    elif type_of_bg == 2:
        diff = cv2.absdiff(bg2.astype("uint8"), image)

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


labels = [
    "blank",
    "down",
    "eight",
    "five",
    "four",
    "left",
    "nine",
    "off",
    "ok",
    "on",
    "one",
    "right",
    "seven",
    "six",
    "three",
    "two",
    "up",
    "zero",
]

# Create the directories
if not os.path.exists("data2"):
    os.makedirs("data2")

for i in labels:
    if not os.path.exists(path2 + i):
        os.makedirs(path2 + i)

total_images = 0
for label in os.listdir(path1):
    images_per_label = 0
    for file in os.listdir(path1 + label):
        total_images += 1
        images_per_label += 1

        src_path = path1 + label + "/" + file
        dest_path = path2 + label + "/" + file

        img = cv2.imread(src_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if label == "blank":
            if images_per_label < 50:
                run_avg1(gray, accumWeight)
            else:
                run_avg2(gray, accumWeight)
        
        if images_per_label < 50:
            hand = segment(gray, type_of_bg=1)
        else:
            hand = segment(gray, type_of_bg=2)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand

            cv2.imwrite(dest_path, thresholded)

    print(label, end=" ")
print("\n", "FINISHED.")

print("\n\nNumber of images:", total_images)

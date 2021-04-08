"""Data Collection for gesture recognition model with OpenCV."""

import cv2
import numpy as np
import os
from image_processing import run_avg, segment

accumWeight = 0.5

# Create the directories
if not os.path.exists("data1"):
    os.makedirs("data1")

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

for i in labels:
    if not os.path.exists("data1/" + i):
        os.makedirs("data1/" + i)


# path
directory = f"data1/"
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1
# initialize num of frames
num_frames = 0
# calibration indicator
calibrated = False

while True:
    ret, frame = cap.read()
    # simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # clone the frame
    clone = frame.copy()

    # getting count of existing images
    count = {
        "zero": len(os.listdir(directory + "zero")),
        "one": len(os.listdir(directory + "one")),
        "two": len(os.listdir(directory + "two")),
        "three": len(os.listdir(directory + "three")),
        "four": len(os.listdir(directory + "four")),
        "five": len(os.listdir(directory + "five")),
        "six": len(os.listdir(directory + "six")),
        "seven": len(os.listdir(directory + "seven")),
        "eight": len(os.listdir(directory + "eight")),
        "nine": len(os.listdir(directory + "nine")),
        "up": len(os.listdir(directory + "up")),
        "down": len(os.listdir(directory + "down")),
        "left": len(os.listdir(directory + "left")),
        "right": len(os.listdir(directory + "right")),
        "off": len(os.listdir(directory + "off")),
        "on": len(os.listdir(directory + "on")),
        "ok": len(os.listdir(directory + "ok")),
        "blank": len(os.listdir(directory + "blank")),
    }

    # printing the count in each set to the screen
    cv2.putText(
        frame,
        "zero : " + str(count["zero"]),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "one : " + str(count["one"]),
        (10, 90),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "two : " + str(count["two"]),
        (10, 110),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "three : " + str(count["three"]),
        (10, 130),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "four : " + str(count["four"]),
        (10, 150),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "five : " + str(count["five"]),
        (10, 170),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "six : " + str(count["six"]),
        (10, 190),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "seven : " + str(count["seven"]),
        (10, 210),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "eight : " + str(count["eight"]),
        (10, 230),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "nine : " + str(count["nine"]),
        (10, 250),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "up : " + str(count["up"]),
        (10, 270),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "down : " + str(count["down"]),
        (10, 290),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "left : " + str(count["left"]),
        (10, 310),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "right : " + str(count["right"]),
        (10, 330),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "off : " + str(count["off"]),
        (10, 350),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "on : " + str(count["on"]),
        (10, 370),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "ok : " + str(count["ok"]),
        (10, 390),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "blank : " + str(count["blank"]),
        (10, 410),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3,
    )

    # coordinates of the Region Of Interest (ROI)
    left = int(0.5 * frame.shape[1])
    top = 10
    right = frame.shape[1] - 10
    bottom = int(0.5 * frame.shape[1])
    # drawing the ROI
    # extracting the ROI
    roi = frame[10:410, 220:520]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successfull...")
    else:
        # segment the hand region
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand

            # draw the segmented region and display the frame
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

            thresholded = cv2.resize(thresholded, (210, 280))
            cv2.imshow("Threshold Image", thresholded)
    
    # the increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(clone, (left, top), (right, bottom), (255, 0, 0), 1)

    cv2.imshow("Data Collection", clone)
    
    # increment the number of frames
    num_frames += 1

    # interrupts
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc
        break
    if interrupt & 0xFF == ord("0"):
        cv2.imwrite(directory + "zero/" + str(count["zero"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("1"):
        cv2.imwrite(directory + "one/" + str(count["one"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("2"):
        cv2.imwrite(directory + "two/" + str(count["two"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("3"):
        cv2.imwrite(directory + "three/" + str(count["three"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("4"):
        cv2.imwrite(directory + "four/" + str(count["four"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("5"):
        cv2.imwrite(directory + "five/" + str(count["five"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("6"):
        cv2.imwrite(directory + "six/" + str(count["six"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("7"):
        cv2.imwrite(directory + "seven/" + str(count["seven"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("8"):
        cv2.imwrite(directory + "eight/" + str(count["eight"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("9"):
        cv2.imwrite(directory + "nine/" + str(count["nine"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("u"):
        cv2.imwrite(directory + "up/" + str(count["up"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("d"):
        cv2.imwrite(directory + "down/" + str(count["down"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("l"):
        cv2.imwrite(directory + "left/" + str(count["left"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("r"):
        cv2.imwrite(directory + "right/" + str(count["right"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("f"):
        cv2.imwrite(directory + "off/" + str(count["off"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("o"):
        cv2.imwrite(directory + "on/" + str(count["on"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("k"):
        cv2.imwrite(directory + "ok/" + str(count["ok"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("b"):
        cv2.imwrite(directory + "blank/" + str(count["blank"]) + ".jpg", roi)

cap.release()
cv2.destroyAllWindows()

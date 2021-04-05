"""Data Collection for gesture recognition model with OpenCV."""

import cv2
import numpy as np
import os

# Create the directories
if not os.path.exists("data1"):
    os.makedirs("data1")

if not os.path.exists("data1/train"):
    os.makedirs("data1/train")

if not os.path.exists("data1/test"):
    os.makedirs("data1/test")

labels = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "up",
    "down",
    "left",
    "right",
    "off",
    "on",
    "ok",
]

for i in labels:
    if not os.path.exists("data1/train/" + i):
        os.makedirs("data1/train/" + i)
    if not os.path.exists("data1/test/" + i):
        os.makedirs("data1/test/" + i)


# train/test mode
mode = "train"
directory = f"data1/{mode}/"
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1

while True:
    ret, frame = cap.read()
    # simulating mirror image
    frame = cv2.flip(frame, 1)

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
    }

    # printing the count in each set to the screen
    cv2.putText(
        frame,
        "zero : " + str(count["zero"]),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "one : " + str(count["one"]),
        (10, 90),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "two : " + str(count["two"]),
        (10, 110),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "three : " + str(count["three"]),
        (10, 130),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "four : " + str(count["four"]),
        (10, 150),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "five : " + str(count["five"]),
        (10, 170),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "six : " + str(count["six"]),
        (10, 190),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "seven : " + str(count["seven"]),
        (10, 210),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "eight : " + str(count["eight"]),
        (10, 230),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "nine : " + str(count["nine"]),
        (10, 250),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "up : " + str(count["up"]),
        (10, 270),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "down : " + str(count["down"]),
        (10, 290),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "left : " + str(count["left"]),
        (10, 310),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "right : " + str(count["right"]),
        (10, 330),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "off : " + str(count["off"]),
        (10, 350),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "on : " + str(count["on"]),
        (10, 370),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )
    cv2.putText(
        frame,
        "ok : " + str(count["ok"]),
        (10, 390),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (227, 132, 54),
        1,
    )

    # coordinates of the Region Of Interest (ROI)
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # drawing the ROI
    # the increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)
    # extracting the ROI
    roi = frame[10:410, 220:520]

    cv2.imshow("Data Collection", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, test_image = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("Threshold Image", test_image)

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
    if interrupt & 0xFF == ord("U"):
        cv2.imwrite(directory + "up/" + str(count["up"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("D"):
        cv2.imwrite(directory + "down/" + str(count["down"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("L"):
        cv2.imwrite(directory + "left/" + str(count["left"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("R"):
        cv2.imwrite(directory + "right/" + str(count["right"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("F"):
        cv2.imwrite(directory + "off/" + str(count["off"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("O"):
        cv2.imwrite(directory + "on/" + str(count["on"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("K"):
        cv2.imwrite(directory + "ok/" + str(count["ok"]) + ".jpg", roi)

cap.release()
cv2.destroyAllWindows()

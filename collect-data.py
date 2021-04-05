import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

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
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/" + i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/" + i)


# Train or test
mode = "test"
directory = "data/test"
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {
        "zero": len(os.listdir(directory + "/zero")),
        "one": len(os.listdir(directory + "/one")),
        "two": len(os.listdir(directory + "/two")),
        "three": len(os.listdir(directory + "/three")),
        "four": len(os.listdir(directory + "/four")),
        "five": len(os.listdir(directory + "/five")),
        "six": len(os.listdir(directory + "/six")),
        "seven": len(os.listdir(directory + "/seven")),
        "eight": len(os.listdir(directory + "/eight")),
        "nine": len(os.listdir(directory + "/nine")),
        "up": len(os.listdir(directory + "/up")),
        "down": len(os.listdir(directory + "/down")),
        "left": len(os.listdir(directory + "/left")),
        "right": len(os.listdir(directory + "/right")),
        "off": len(os.listdir(directory + "/off")),
        "on": len(os.listdir(directory + "/on")),
        "ok": len(os.listdir(directory + "/ok")),
    }

    # Printing the count in each set to the screen
    # cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "IMAGE COUNT", (10, ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(
        frame,
        "ZERO : " + str(count["zero"]),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "ONE : " + str(count["one"]),
        (10, 80),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "TWO : " + str(count["two"]),
        (10, 90),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "THREE : " + str(count["three"]),
        (10, 180),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "FOUR : " + str(count["four"]),
        (10, 200),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "FIVE : " + str(count["five"]),
        (10, 220),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "SIX : " + str(count["six"]),
        (10, 230),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "seven : " + str(count["seven"]),
        (10, 100),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "eight : " + str(count["eight"]),
        (10, 110),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "nine : " + str(count["nine"]),
        (10, 120),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "up : " + str(count["up"]),
        (10, 130),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "down : " + str(count["down"]),
        (10, 140),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "left : " + str(count["left"]),
        (10, 150),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "right : " + str(count["right"]),
        (10, 160),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "off : " + str(count["off"]),
        (10, 170),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    # cv2.putText(frame, "i : "+str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(
        frame,
        "on : " + str(count["on"]),
        (10, 190),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "ok : " + str(count["ok"]),
        (10, 200),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 255),
        1,
    )

    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]
    #    roi = cv2.resize(roi, (64, 64))

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, test_image = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # time.sleep(5)
    # cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    # test_image = func("/home/rc/Downloads/soe/im1.jpg")

    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("test", test_image)

    # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(mask, kernel, iterations=1)
    # img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    #    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    ##gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow("GrayScale", gray)
    ##blur = cv2.GaussianBlur(gray,(5,5),2)

    # blur = cv2.bilateralFilter(roi,9,75,75)

    ##th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ##ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow("ROI", roi)
    # roi = frame
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord("0"):
        cv2.imwrite(directory + "/zero/" + str(count["zero"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("1"):
        cv2.imwrite(directory + "/one/" + str(count["one"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("2"):
        cv2.imwrite(directory + "/two/" + str(count["two"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("3"):
        cv2.imwrite(directory + "/three/" + str(count["three"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("4"):
        cv2.imwrite(directory + "/four/" + str(count["four"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("5"):
        cv2.imwrite(directory + "/five/" + str(count["five"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("6"):
        cv2.imwrite(directory + "/six/" + str(count["six"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("7"):
        cv2.imwrite(directory + "/seven/" + str(count["seven"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("8"):
        cv2.imwrite(directory + "/eight/" + str(count["eight"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("9"):
        cv2.imwrite(directory + "/nine/" + str(count["nine"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("U"):
        cv2.imwrite(directory + "/up/" + str(count["up"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("D"):
        cv2.imwrite(directory + "/down/" + str(count["down"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("L"):
        cv2.imwrite(directory + "/left/" + str(count["left"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("R"):
        cv2.imwrite(directory + "/right/" + str(count["right"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("F"):
        cv2.imwrite(directory + "/off/" + str(count["off"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("O"):
        cv2.imwrite(directory + "/on/" + str(count["on"]) + ".jpg", roi)
    if interrupt & 0xFF == ord("K"):
        cv2.imwrite(directory + "/ok/" + str(count["ok"]) + ".jpg", roi)

cap.release()
cv2.destroyAllWindows()
"""
up = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(up):
    for file in walk[2]:
        roi = cv2.imread(up+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""

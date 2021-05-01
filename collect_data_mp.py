import cv2
import numpy as np
import os
from image_processing import detectHands, getHands

# path
directory = "mp_data/"

# training labels
labels = ["zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine",
          "up", "down", "left", "right", "off", "on",
          "ok"]

# create the directories
if not os.path.exists("mp_data"):
    os.makedirs("mp_data")

for i in labels:
    if not os.path.exists(directory + i):
        os.makedirs(directory + i)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # removing mirror image
    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    # blank image
    blank_image = np.zeros((height, width, channels), np.uint8)

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
    cv2.putText(frame, "ZERO : " + str(count["zero"]), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "ONE : " + str(count["one"]), (10, 90),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "TWO : " + str(count["two"]), (10, 110),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "THREE : " + str(count["three"]), (10, 130),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "FOUR : " + str(count["four"]), (10, 150),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "FIVE : " + str(count["five"]), (10, 170),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "SIX : " + str(count["six"]), (10, 190),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "SEVEN : " + str(count["seven"]), (10, 210),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "EIGHT : " + str(count["eight"]), (10, 230),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "NINE : " + str(count["nine"]), (10, 250),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "UP : " + str(count["up"]), (10, 270),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "DOWN : " + str(count["down"]), (10, 290),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "LEFT : " + str(count["left"]), (10, 310),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "RIGHT : " + str(count["right"]), (10, 330),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "OFF : " + str(count["off"]), (10, 350),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "ON : " + str(count["on"]), (10, 370),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(frame, "OK : " + str(count["ok"]), (10, 390),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    
    # detect mediapipe hands    
    mediapipe_hands = detectHands(frame, blank_image)
    if type(mediapipe_hands) != type(None):
        cv2.imshow("Mediapipe Hands", mediapipe_hands)
    else:
        cv2.putText(frame, "NO HANDS", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # data collection frame
    cv2.imshow("Data Collection", frame)

    # on keypress
    keypress = cv2.waitKey(10) & 0xFF
    if keypress == 27:  # esc
        break

    # take pictures
    if keypress == ord("0"):
        cv2.imwrite(directory + "zero/" + str(count["zero"]) + ".jpg", mediapipe_hands)
    if keypress == ord("1"):
        cv2.imwrite(directory + "one/" + str(count["one"]) + ".jpg", mediapipe_hands)
    if keypress == ord("2"):
        cv2.imwrite(directory + "two/" + str(count["two"]) + ".jpg", mediapipe_hands)
    if keypress == ord("3"):
        cv2.imwrite(directory + "three/" + str(count["three"]) + ".jpg", mediapipe_hands)
    if keypress == ord("4"):
        cv2.imwrite(directory + "four/" + str(count["four"]) + ".jpg", mediapipe_hands)
    if keypress == ord("5"):
        cv2.imwrite(directory + "five/" + str(count["five"]) + ".jpg", mediapipe_hands)
    if keypress == ord("6"):
        cv2.imwrite(directory + "six/" + str(count["six"]) + ".jpg", mediapipe_hands)
    if keypress == ord("7"):
        cv2.imwrite(directory + "seven/" + str(count["seven"]) + ".jpg", mediapipe_hands)
    if keypress == ord("8"):
        cv2.imwrite(directory + "eight/" + str(count["eight"]) + ".jpg", mediapipe_hands)
    if keypress == ord("9"):
        cv2.imwrite(directory + "nine/" + str(count["nine"]) + ".jpg", mediapipe_hands)
    if keypress == ord("u"):
        cv2.imwrite(directory + "up/" + str(count["up"]) + ".jpg", mediapipe_hands)
    if keypress == ord("d"):
        cv2.imwrite(directory + "down/" + str(count["down"]) + ".jpg", mediapipe_hands)
    if keypress == ord("l"):
        cv2.imwrite(directory + "left/" + str(count["left"]) + ".jpg", mediapipe_hands)
    if keypress == ord("r"):
        cv2.imwrite(directory + "right/" + str(count["right"]) + ".jpg", mediapipe_hands)
    if keypress == ord("f"):
        cv2.imwrite(directory + "off/" + str(count["off"]) + ".jpg", mediapipe_hands)
    if keypress == ord("o"):
        cv2.imwrite(directory + "on/" + str(count["on"]) + ".jpg", mediapipe_hands)
    if keypress == ord("k"):
        cv2.imwrite(directory + "ok/" + str(count["ok"]) + ".jpg", mediapipe_hands)

cap.release()
cv2.destroyAllWindows()

"""Visualizing the results."""

from keras.models import load_model
import cv2
import numpy as np
from image_processing import run_avg, segment

accumWeight = 0.5


def _load_weights():
    """Load Model Weights."""
    try:
        model = load_model("model/model.h5")
        print(model.summary())
        return model

    except Exception as e:
        return None


def getPredictedClass(model):
    image = cv2.imread('Temp.png')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 100))
    gray_image = gray_image.reshape(1, 100, 100, 1)

    prediction = model.predict_on_batch(gray_image)

    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "BLANK"
    elif predicted_class == 1:
        return "DOWN"
    elif predicted_class == 2:
        return "EIGHT"
    elif predicted_class == 3:
        return "FIVE"
    elif predicted_class == 4:
        return "FOUR"
    elif predicted_class == 5:
        return "LEFT"
    elif predicted_class == 6:
        return "NINE"
    elif predicted_class == 7:
        return "OFF"
    elif predicted_class == 8:
        return "OK"
    elif predicted_class == 9:
        return "ON"
    elif predicted_class == 10:
        return "ONE"
    elif predicted_class == 11:
        return "RIGHT"
    elif predicted_class == 12:
        return "SEVEN"
    elif predicted_class == 13:
        return "SIX"
    elif predicted_class == 14:
        return "THREE"
    elif predicted_class == 15:
        return "TWO"
    elif predicted_class == 16:
        return "ONE"
    elif predicted_class == 17:
        return "UP"
    elif predicted_class == 18:
        return "ZERO"


if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 310, 310, 610

    # initialize num of frames
    num_frames = 0
    # calibration indicator
    calibrated = False

    model = _load_weights()

    while True:
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        # frame = cv2.resize(frame, (500, 500))
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

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
                cv2.drawContours(
                    clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                # fingers = count(thresholded, segmented)
                cv2.imwrite('Temp.png', thresholded)
                predictedClass = getPredictedClass(model)
                cv2.putText(clone, str(predictedClass), (70, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Gesture Recognition", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        elif keypress == ord("c"):
            num_frames = 0

    # free up memory
    camera.release()
    cv2.destroyAllWindows()

import numpy as np
import cv2
import os
from image_processing import func

# Create the directories
if not os.path.exists("data2"):
    os.makedirs("data2")

if not os.path.exists("data2/train"):
    os.makedirs("data2/train")

if not os.path.exists("data2/test"):
    os.makedirs("data2/test")

path1 = "data1/"
path2 = "data2/"

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

c1 = 0
c2 = 0

for directory in os.listdir(path1):
    for label in os.listdir(path1 + directory):
        if not os.path.exists(path2 + "train/" + label):
            os.makedirs(path2 + "train/" + label)
        if not os.path.exists(path2 + "test/" + label):
            os.makedirs(path2 + "test/" + label)

        for file in os.listdir(path1 + directory + "/" + label):
            actual_path = path1 + directory + "/" + label + "/" + file
            dest_path = path2 + directory + "/" + label + "/" + file

            if directory == "train":
                c1 += 1
            else:
                c2 += 1

            img = cv2.imread(actual_path, 0)
            bw_image = func(actual_path)
            cv2.imwrite(dest_path, bw_image)
        print(label, end=" ")
    print("\n", directory, "FINISHED.")

print("\n\nNumber of trianing images:", c1)
print("Number of test images:", c2)

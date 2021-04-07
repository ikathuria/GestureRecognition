import numpy as np
import cv2
import os
from image_processing import preprocess


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

path1 = "data1/"
path2 = "data2/"

# Create the directories
if not os.path.exists("data2"):
    os.makedirs("data2")

for i in labels:
    if not os.path.exists(path2 + i):
        os.makedirs(path2 + i)


i = 0
for label in os.listdir(path1):
    for file in os.listdir(path1 + label):
        i += 1
        src_path = path1 + label + "/" + file
        dest_path = path2 + label + "/" + file

        img = cv2.imread(src_path, 0)
        bw_image = preprocess(src_path)

        cv2.imwrite(dest_path, bw_image)

    print(label, end=" ")
print("\n", "FINISHED.")

print("\n\nNumber of images:", i)

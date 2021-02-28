import tensorflow as tf
import cv2 as c
import scipy as s
import sys
import os
import unloader
import numpy as np

labelKey = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'z', 'y', 'z',]

def main():
    if (len(sys.argv) != 2):
        sys.exit("Incorrect Usage.\nUsage: main.py fileName")
    if not os.path.exists(sys.argv[1]):
        raise ReferenceError("File given not found")
    u = unloader.ai()
    ai = u.model
    print("loaded ai")

    sizeX, sizeY, imgs = u.unloadImg(sys.argv[1])

    for x in range(sizeX):
        line = ""
        for y in range(sizeY):
            prediction = ai.predict(np.reshape(imgs[x][y], (1, 28, 28, 1))).argmax()
            if (prediction is not None):
                line += labelKey[prediction]
            else:
                line += " "
        print(line)

if __name__ == "__main__":
    main()
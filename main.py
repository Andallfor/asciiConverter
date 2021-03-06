import tensorflow as tf
import cv2 as c
import scipy as s
import sys
import os
import unloader
import numpy as np
import emnist

labelKey = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'z', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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
            p = ai.predict(imgs[x][y].reshape((1, 28, 28, 1)))
            prediction = p.argmax()
            
            # L is the default value so uh
            # blame the dataset not having blank spaces as a training test
            # not me
            # shhhhhhhh
            if (labelKey[prediction] is not 'L'):
                line += str(labelKey[prediction])
            else:
                line += " "
        print(line)

if __name__ == "__main__":
    main()
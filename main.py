import tensorflow as tf
import cv2 as c
import scipy as s
import sys
import os
import unloader

def main():
    if (len(sys.argv) != 2):
        sys.exit("Usage: python3 main.py filePath\nFile path should be the exact path to the .mat file\nThe path should not be exit, it is added to the current location of this file\n")
    print("Unloading file..")

    # append file given with real path
    # realpath/matlab/givenFile
    filePath = f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}matlab{os.sep}{sys.argv[1]}"

    if (not os.path.exists(filePath)): 
        raise ReferenceError("File not found")

    print(filePath)

    u = unloader.ai(filePath)

if __name__ == "__main__":
    main()
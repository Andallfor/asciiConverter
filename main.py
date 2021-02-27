import tensorflow as tf
import cv2 as c
import scipy as s
import sys
import os
import unloader

def main():
    u = unloader.ai()
    ai = u.model
    print("loaded ai")

if __name__ == "__main__":
    main()
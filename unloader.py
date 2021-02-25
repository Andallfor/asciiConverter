import tensorflow as tf
import cv2 as c
import scipy as s
import scipy.io as si
import sys
# where the .mat files are unloaded
# and an ai class is returned

class ai():
    def __init__(self, file):
        self.defaultFile = file

        # unload
        try:
            self.matFile = si.loadmat(self.defaultFile, chars_as_strings = True)
        except:
            raise TypeError("Incorrect file format, must be .mat")
        
        print(self.matFile['dataset'])

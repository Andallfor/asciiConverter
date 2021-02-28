import tensorflow as tf
import cv2 as c
import scipy as s
import scipy.io as si
import sys
import numpy as np
import emnist
import os
import image

class ai():
    def __init__(self):
        print(os.path.dirname(os.path.abspath(__file__)) + os.sep + "saved_model.pb")
        if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + os.sep + "saved_model.pb"):
            # retrieve ai
            self.model = tf.keras.models.load_model(os.path.dirname(os.path.abspath(__file__)) + os.sep)
            print("found previous ai")
        else:
            # generate ai
            self.generate()
    
    def generate(self):
        self.trainImages, self.traingLabels = emnist.extract_training_samples('balanced')
        self.testImages, self.testLabels = emnist.extract_test_samples('balanced')

        print(self.trainImages.shape)

        self.trainImages, self.testImages = self.trainImages / 255.0, self.testImages / 255.0
        print(self.trainImages.shape)
        self.traingLabels = tf.keras.utils.to_categorical(self.traingLabels)
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels)
        
        self.trainImages = self.trainImages.reshape(
            self.trainImages.shape[0], self.trainImages.shape[1], self.trainImages.shape[2], 1)
        print(self.trainImages.shape)
        self.testImages = self.testImages.reshape(
            self.testImages.shape[0], self.testImages.shape[1], self.testImages.shape[2], 1)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                80, (3, 3), activation="relu", input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(47, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        self.model.fit(self.trainImages, self.traingLabels, epochs=7)
        self.model.evaluate(self.testImages,  self.testLabels, verbose=2)

        self.model.save(os.path.dirname(os.path.abspath(__file__)))
    
    def unloadImg(self, path):
        '''
        returns sizeX, sizeY, [x][y]

        [x][y] gives (28, 28, 1)
        '''
        # load img in black and white
        img = c.imread(path, c.IMREAD_GRAYSCALE)

        # blur
        img = c.GaussianBlur(img, (5, 5), 0)

        # get size, increments of 28 (convert to int to floor it)
        sizeX = int(img.shape[0] / 28)
        sizeY = int(img.shape[1] / 28)

        returnArray = [[] for i in range(sizeX)]

        for x in range(0, sizeX * 27, 28):
            for y in range(0, sizeY * 27, 28):
                s = img[x : x + 28, y : y + 28]
                returnArray[int(x/28)].append(s)

        return sizeX, sizeY, returnArray
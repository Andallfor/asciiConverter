import tensorflow as tf
import cv2 as c
import scipy as s
import scipy.io as si
import sys
import numpy as np
import emnist

class ai():
    def __init__(self):
        self.trainImages, self.traingLabels = emnist.extract_training_samples('balanced')
        self.testImages, self.testLabels = emnist.extract_test_samples('balanced')

        self.trainImages, self.testImages = self.trainImages / 255.0, self.testImages / 255.0
        self.traingLabels = tf.keras.utils.to_categorical(self.traingLabels)
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels)
        
        self.trainImages = self.trainImages.reshape(
            self.trainImages.shape[0], self.trainImages.shape[1], self.trainImages.shape[2], 1)
        self.testImages = self.testImages.reshape(
            self.testImages.shape[0], self.testImages.shape[1], self.testImages.shape[2], 1)
            
        print(self.trainImages.shape)
        print(self.traingLabels.shape)
        print(self.testImages.shape)
        print(self.testLabels.shape)


        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(47, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        self.model.fit(self.trainImages, self.traingLabels, epochs=10)
        self.model.evaluate(self.testImages,  self.testLabels, verbose=2)
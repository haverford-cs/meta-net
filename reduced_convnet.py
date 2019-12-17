"""
Smaller version of the convolutional neural network architecture that I used
in the class competition.
Authors: Gareth Nicholas
Date: December 9th, 2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, \
    BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import Model, regularizers

from tensorflow.keras.models import Sequential

class reduced_convnet(Model):

    def __init__(self):
        self.custom_model_name = "reduced convnet"
        super(reduced_convnet, self).__init__()
        self.model_name = "reduced_convnet"
        shape = (32, 32, 3)
        model = Sequential()

        # Similar to the original convnet
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001),
            input_shape = shape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001)
            , padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        # End with this convolutional block, in convnet() there is
        # one extra block
        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.01),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.01),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        # And the classification...
        model.add(Flatten())
        # Drop dense layers from 4000 nodes to 512. Saves a lot of parameters.
        model.add(Dense(512, activation = tf.nn.relu))
        model.add(Dense(512, activation = tf.nn.relu))
        model.add(Dense(43, activation = tf.nn.softmax))

        self.model = model

    def call(self, x):
        return self.model(x)

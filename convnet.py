"""
Convolutional neural network architecture that I used in the class competition.
The starting benchmark.
Authors: Gareth Nicholas
Date: December 8th, 2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, \
    BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import Model, regularizers

# I actually want to move away from sequential for some future model
# iterations...
from tensorflow.keras.models import Sequential

# This model is massive and probably terrible if we want to reduce the overall
# size, leaving it here as a placeholder.
class convnet(Model):

    def __init__(self):
        super(convnet, self).__init__()
        shape = (32, 32, 3)
        model = Sequential()

        # Some convolutional block stuff
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001),
            input_shape = shape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001)
            , padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.01),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.01),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (2, 2), kernel_regularizer=regularizers.l2(0.1),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv2D(256, (2, 2), kernel_regularizer=regularizers.l2(0.1),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        # And the classification...
        model.add(Flatten())
        model.add(Dense(4000, activation = tf.nn.relu))
        model.add(Dense(4000, activation = tf.nn.relu))
        model.add(Dense(43, activation = tf.nn.softmax))

        self.model = model

    def call(self, x):
        # Apply convolutional layers
        # Then flatten
        # Then output probabilities
        return self.model(x)

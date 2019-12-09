"""
Convolutional neural network architecture which uses multi-scale features.
Authors: Gareth Nicholas
Date: December 9th, 2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, \
    BatchNormalization, MaxPooling2D, Dropout, concatenate
from tensorflow.keras import Model, regularizers, Input


class multi_scale_conv(Model):

    def __init__(self):
        super(multi_scale_conv, self).__init__()
        img_shape = (32, 32, 3)

        in_layer = Input(shape = img_shape)
        conv1 = Conv2D(32, (5, 5), kernel_regularizer=regularizers.l2(1e-4),
        padding = "same")(in_layer)
        act1 = Activation("relu")(conv1)
        pool1 = MaxPooling2D((2,2))(act1)
        conv2 = Conv2D(64, (4, 4), kernel_regularizer=regularizers.l2(1e-4),
        padding = "same")(pool1)
        act2 = Activation("relu")(conv2)
        pool2 = MaxPooling2D((2,2))(act2)
        conv3 = Conv2D(128, (4, 4), kernel_regularizer=regularizers.l2(1e-4),
        padding = "same")(pool2)
        act3 = Activation("relu")(conv3)
        pool3 = MaxPooling2D((2,2))(act3)

        scale_pool1 = MaxPooling2D((4,4))(pool1)
        scale_pool2 = MaxPooling2D((2,2))(pool2)

        flatten1 = Flatten()(scale_pool1)
        flatten2 = Flatten()(scale_pool2)
        flatten3 = Flatten()(pool3)

        combined = concatenate([flatten1, flatten2, flatten3])
        dense1 = Dense(1024, activation = tf.nn.relu)(combined)
        drop1 = Dropout(0.3)(dense1)
        dense2 = Dense(1024, activation = tf.nn.relu)(dense1)
        drop2 = Dropout(0.3)(dense2)
        dense3 = Dense(43, activation = tf.nn.softmax)(drop2)

        self.model = Model(inputs=[in_layer], outputs=[dense3])


    def call(self, x):
        # Apply convolutional layers
        # Then flatten
        # Then output probabilities
        return self.model(x)
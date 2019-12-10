"""
Several small networks used for plotting size of network vs. accuracy.
Authors: Gareth Nicholas
Date: December 9th, 2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, \
    BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import Model, regularizers

from tensorflow.keras.models import Sequential

class reduced_dense_256(Model):
    """Reduce the number of nodes in the dense layers to 256.

    Attributes
    ----------
    model : tf model
        The underlying model.

    """

    def __init__(self):
        super(reduced_dense_256, self).__init__()
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

        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        # And the classification...
        model.add(Flatten())
        model.add(Dense(256, activation = tf.nn.relu))
        model.add(Dense(256, activation = tf.nn.relu))
        model.add(Dense(43, activation = tf.nn.softmax))

        self.model = model

    def call(self, x):
        return self.model(x)

class reduced_dense_256_pool(Model):
    """Reduce the number of nodes in the dense layers to 256. Increase in
    number of pooling operations.

    Attributes
    ----------
    model : tf model
        The underlying model.

    """

    def __init__(self):
        super(reduced_dense_256_pool, self).__init__()
        shape = (32, 32, 3)
        model = Sequential()

        # Some convolutional block stuff
        model.add(Conv2D(32, (5, 5), kernel_regularizer=regularizers.l2(0.0001),
            input_shape = shape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((5, 5)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (5, 5), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(0.001),
            padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        # And the classification...
        model.add(Flatten())
        model.add(Dense(256, activation = tf.nn.relu))
        model.add(Dense(256, activation = tf.nn.relu))
        model.add(Dense(43, activation = tf.nn.softmax))

        self.model = model

    def call(self, x):
        return self.model(x)

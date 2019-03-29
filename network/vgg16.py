from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
import cv2
import numpy as np


class VGG_16:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # 如果为 channels first, 调整 input shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # Block 1
        # model.add(ZeroPadding2D((1, 1), input_shape=inputShape))
        model.add(Conv2D(64, (3, 3),
                         input_shape=inputShape,
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))

        # Block 2
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))

        # Block 3
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))

        # Block 4
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))

        # Block 5
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model

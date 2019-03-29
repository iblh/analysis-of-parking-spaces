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
                         padding="same",
                         name='block1_conv1'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               name='block1_pool'))

        # Block 2
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block2_conv1'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               name='block2_pool'))

        # Block 3
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block3_conv1'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block3_conv2'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block3_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               name='block3_pool'))

        # Block 4
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block4_conv1'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block4_conv2'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block4_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               name='block4_pool'))

        # Block 5
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block5_conv1'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block5_conv2'))
        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding="same",
                         name='block5_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               name='block5_pool'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model

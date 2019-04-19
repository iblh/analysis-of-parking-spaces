from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # 如果为 channels first, 调整 input shape 
        # 和 channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block 1  CONV => RELU => POOL
        model.add(Conv2D(filters=24, input_shape=inputShape, kernel_size=(11, 11),
                         strides=(4, 4), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='same'))

        # Block 2  CONV => RELU => POOL
        model.add(Conv2D(filters=64,
                         kernel_size=(11, 11), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='same'))

        # Block 3  CONV => RELU
        model.add(Conv2D(filters=96,
                         kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))

        # Block 4  CONV => RELU
        model.add(Conv2D(filters=96,
                         kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))

        # Block 5  CONV => RELU => POOL
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='same'))

        # Passing it to a dense layer
        model.add(Flatten())
        # FC => RELU Layer
        model.add(Dense(1024))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))

        # FC => RELU layer
        model.add(Dense(1024))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # FC => RELU layer
        model.add(Dense(256))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # FC => softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

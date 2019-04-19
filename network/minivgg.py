
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras import regularizers
from keras import backend as K


class MiniVGG:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # 'channels last' and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # 如果为 channels first, 调整 input shape
        # 和 channels dimension
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # Block 1  CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # Block 2  (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 3  (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Passing it to a dense layer
        model.add(Flatten())
        # FC => RELU layers
        model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # FC => softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential


class NeuralNetwork:
    BATCH_SIZE = 32

    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape x's to have a single channel (grayscale) and one-hot encode y's
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32) / 255.0
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype(np.float32) / 255.0
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)
        self.model = None

    def build_model(self):
        input_shape = (28, 28, 1)
        model = Sequential()
        model.add(Input(input_shape))
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation(activation='softmax'))
        self.model = model

    def train_model(self):
        self.build_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=NeuralNetwork.BATCH_SIZE,
                       epochs=10,
                       validation_data=(self.x_test, self.y_test))
        self.model.save('digits_cnn.h5')

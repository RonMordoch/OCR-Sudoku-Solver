import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2


CONV_KERNEL = (3,3)

mnist = tf.keras.datasets.mnist
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

BATCH_SIZE = 32


class NeuralNetwork:

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape x's to have a single channel (grayscale) and one-hot encode y's
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32) / 255.0
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype(np.float32) / 255.0
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

    def build_model(self):
        input_shape = (28,28,1)
        model = Sequential()
        model.add(Input(input_shape))
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(10))
        model.add(Activation(activation='softmax'))
        return model

    def evaluate_model(self):
        model = self.build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=self.x_train, y=self.y_train,batch_size=BATCH_SIZE, epochs=10, validation_data=(self.x_test, self.y_test))
        model.save('digits_cnn.h5')


    def load_saved_model(self):
        model = load_model('digits_cnn.h5')
        sample = self.x_train[258]
        while True:
            cv2.imshow('sample', sample)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        sample = sample.reshape((1,28,28,1))
        digit = model.predict_classes(sample)
        print(digit)




cnn = NeuralNetwork()
m = cnn.load_saved_model()

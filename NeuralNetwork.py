import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class NeuralNetwork:
    BATCH_SIZE = 32
    EPOCHS = 10

    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape x's to have a single channel (grayscale) , normalize
        self.x_train = x_train.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        self.x_test = x_test.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        # hot-encode y's
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)
        # split training into train/validation set , take 10% as validation
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_train,
                                                                                  self.y_train,
                                                                                  test_size=0.1)
        self.datagen = ImageDataGenerator(
            rotation_range=10,  # degree range for random rotations
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            zoom_range=0.1,  # range for random zoom
            shear_range=0.1  # shear intensity (angle, counter-clockwise direction, degrees)
        )
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(28, 28, 1)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(Activation(activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(Activation(activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3)))
        model.add(Activation(activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3)))
        model.add(Activation(activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3)))
        model.add(Activation(activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(Activation(activation="relu"))
        model.add(Dense(10))
        model.add(Activation(activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model = model

    def train_model(self):
        self.build_model()
        # create data augmentation generators
        train_gen = self.datagen.flow(self.x_train, self.y_train, batch_size=self.BATCH_SIZE)
        valid_gen = self.datagen.flow(self.x_valid, self.y_valid, batch_size=self.BATCH_SIZE)

        self.model.fit(x=train_gen,
                       epochs=self.EPOCHS,
                       steps_per_epoch=self.x_train.shape[0] // self.BATCH_SIZE,
                       validation_data=valid_gen,
                       validation_steps=self.x_valid.shape[0] // self.BATCH_SIZE)
        self.model.save('digits_cnn_v2.h5')

    def evaluate_model(self):
        test_gen = self.datagen.flow(self.x_test, self.y_test, batch_size=self.BATCH_SIZE)
        results = self.model.evaluate(x=test_gen)
        print("test loss, test acc:", results)

    def load_model(self):
        self.model = tf.keras.models.load_model('digits_cnn_v2.h5')


cnn = NeuralNetwork()
# run once:
cnn.train_model()
cnn.load_model()
cnn.evaluate_model()
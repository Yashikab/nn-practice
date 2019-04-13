# coding utf-8

import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.preprocessing import normalize

## Importing the MNIST dataset using Keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape for vector input
N, x, y = X_train.shape
X_train = normalize(np.reshape(X_train, (N, x * y)))

N, x, y = X_test.shape
X_test = normalize(np.reshape(X_test, (N, x * y)))

# one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(output_dim=750, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=1)

## Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
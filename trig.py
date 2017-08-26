# predict a simple trig function with deep learning

import numpy as np
import keras
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000
NUM_TRAIN_SAMPLES = int(NUM_SAMPLES * 0.75)
batch_size = 10
epochs = 1000

def func(x):
    return np.sin(x) + np.random.randn(*x.shape)*0.01

x = np.random.rand(NUM_SAMPLES, 1) * 2*np.pi
y = func(x)

x_train, y_train = x[:NUM_TRAIN_SAMPLES], y[:NUM_TRAIN_SAMPLES]
x_test, y_test = x[NUM_TRAIN_SAMPLES:], y[NUM_TRAIN_SAMPLES:]



def create_network():
    x_in = keras.Input(shape=[1])
    x = keras.layers.Dense(100, activation='elu')(x_in)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=x_in, outputs=x)
    model.compile(keras.optimizers.SGD(), loss='mse', metrics=['mse'])
    return model

model = create_network()
model.fit(x=x_train, y=y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, shuffle=True)

pred = model.predict(x_test)
fig, ax = plt.subplots()
ax.scatter(x_test, pred, color='r')
ax.scatter(x_test, y_test, color='b')
plt.show()
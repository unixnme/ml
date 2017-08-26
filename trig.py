# predict a simple trig function with deep learning

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K
import os


np.random.seed(0)

NUM_SAMPLES = 1000
FREQ = 1
NUM_TRAIN_SAMPLES = int(NUM_SAMPLES * 0.75)

batch_size = 10
epochs = 10
model_path = 'model.h5'

def func(x):
    return np.sin(x) ** 2 + np.random.randn(*x.shape)*0.01

def relu():
    return keras.layers.Lambda(lambda x: K.maximum(0., x))

def softplus(alpha):
    return keras.layers.Lambda(lambda x: 1./alpha * K.log(1 + K.exp(alpha*x)))

x = (np.random.rand(NUM_SAMPLES, 1) - 0.5) * 2*np.pi * FREQ
y = func(x)

x_train, y_train = x[:NUM_TRAIN_SAMPLES], y[:NUM_TRAIN_SAMPLES]
x_test, y_test = x[NUM_TRAIN_SAMPLES:], y[NUM_TRAIN_SAMPLES:]

def create_network(hidden_units, activation):
    x_in = keras.Input(shape=[1])
    x = x_in
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=activation)(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=x_in, outputs=x)
    model.compile(keras.optimizers.SGD(), loss='mse', metrics=['mse'])
    return model

hidden_layers = [10,10]
activations = ['relu', softplus(10.), softplus(5.), softplus(2.), softplus(1.)]
labels = ['relu', 'softplus10', 'softplus5', 'softplus2', 'softplus1']
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
if os.path.isfile(model_path):
    os.remove(model_path)
for activation, label in zip(activations, labels):
    np.random.seed(0)
    model = create_network(hidden_layers, activation)
    if os.path.isfile(model_path):
        model.load_weights(model_path)
    history = model.fit(verbose=False, x=x_train, y=y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, shuffle=True)

    pred = model.predict(x_test)
    ax1.scatter(x_test, pred, label=label, marker='.')
    ax2.plot(history.history['loss'], label=label)
    model.save_weights(model_path)

ax1.scatter(x_test, y_test, label='true', marker='.')
ax1.legend()
ax2.legend()
plt.show()
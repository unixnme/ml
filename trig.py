# predict a simple trig function with deep learning

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K
import os
from keras.layers.advanced_activations import LeakyReLU

np.random.seed(0)

NUM_SAMPLES = 10
FREQ = 0.5
NUM_TRAIN_SAMPLES = int(NUM_SAMPLES * 1)

batch_size = 10
epochs = 1000
model_path = 'model.h5'

def func(x, noise=True):
    return x + noise * np.random.randn(*x.shape)*0.1


def relu():
    return keras.layers.Lambda(lambda x: K.maximum(0., x))

def softplus(alpha):
    return keras.layers.Lambda(lambda x: 1./alpha * K.log(1 + K.exp(alpha*x)))

#x = (np.random.rand(NUM_SAMPLES, 1) - 0.5) * 2*np.pi * FREQ
x = (np.arange(0, 1, .1) - 0.5) * 2*np.pi * FREQ
x_draw = (np.arange(0, 1, .01) - 0.5) * 2*np.pi * FREQ
y = func(x)
y_draw = func(x_draw, False)

x_train, y_train = x[:NUM_TRAIN_SAMPLES], y[:NUM_TRAIN_SAMPLES]
x_test, y_test = x[NUM_TRAIN_SAMPLES:], y[NUM_TRAIN_SAMPLES:]

def create_network(hidden_units, activation):
    x_in = keras.Input(shape=[1])
    x = x_in
    for units in hidden_units:
        x = keras.layers.Dense(units, kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = activation(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=x_in, outputs=x)
    #model.compile(keras.optimizers.SGD(momentum=0.9, decay=1e-6, nesterov=True), loss='mse', metrics=['mse'])
    model.compile(keras.optimizers.Adam(), loss='mse', metrics=['mse'])
    return model

hidden_layers = [100,100]
activations = [keras.layers.Activation('relu'), softplus(alpha=10), softplus(5), softplus(4), softplus(3), softplus(2), softplus(1),
               softplus(0.5), softplus(0.25)]
labels = ['relu', 'softplus10', 'softplus5', 'softplus4', 'softplus3', 'softplus2', 'softplus1', 'softplus0.5',
          'softplus0.25']
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

    pred = model.predict(x_draw)
    ax1.plot(x_draw, pred, '--', label=label)
    ax2.plot(history.history['loss'], label=label)
    model.save_weights(model_path)

ax1.scatter(x_train, y_train, label='train', marker='.')
ax1.plot(x_draw, y_draw, '--', label='theory')
ax1.legend()
ax2.legend()
plt.show()
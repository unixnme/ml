import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 2
np.random.seed(0)
NUM_SAMPLES = 1000

def create_model(dense_units, activation, regularizer):
    x_in = keras.layers.Input(shape=(2,))
    x = x_in
    for units in dense_units:
        x = keras.layers.Dense(units, kernel_regularizer=regularizer)(x)
        if activation in ['relu', 'elu', 'sigmoid']:
            x = keras.layers.Activation(activation)(x)
        else:
            x = keras.layers.Lambda(lambda x: 1/activation * K.log(1 + K.exp(activation * x)))(x)
    x = keras.layers.Dense(NUM_CLASSES, kernel_regularizer=regularizer)(x)
    x = keras.layers.Activation('softmax')(x)
    model = keras.models.Model(inputs=x_in, outputs=x)
    return model


x = np.random.rand(NUM_SAMPLES, 2) * 2.5 - 1.25
y = np.zeros(shape=(NUM_SAMPLES,)).astype(np.int32)

for idx in range(NUM_SAMPLES):
    if np.sum(x[idx]**2) + (np.random.rand() - 0.5) * 0.2 >= 1:
        pass
    else:
        y[idx] = 1



model = create_model(dense_units=[1], activation='elu', regularizer=keras.regularizers.l2(1e-3))
model.compile(optimizer=keras.optimizers.SGD(momentum=0.9, decay=1e-6, nesterov=True),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
hist = model.fit(x=x, y=y.reshape(-1,1), batch_size=20, epochs=100, shuffle=True)
pred = np.argmax(model.predict(x), axis=-1)
print "\n"
print model.evaluate(x, y)
print

print model.layers[1].get_weights()
print model.layers[3].get_weights()

model.save('circle.h5')

plt.scatter(x[pred==0, 0], x[pred==0, 1], color="b")
plt.scatter(x[pred==1, 0], x[pred==1, 1], color="r")
plt.show()


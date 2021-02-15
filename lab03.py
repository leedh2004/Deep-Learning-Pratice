import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = [1,2,3,4]
Y = [0, -1, -2, -3]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)
model.summary()
history = model.fit(X, Y, epochs=100)
y_predict = model.predict(np.array([5,4]))
print(y_predict)
print(history.history)
plt.plot(history.history['loss'])
plt.show()
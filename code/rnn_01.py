#!/usr/bin/env python

from custom_functs import *
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)
# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print("X_train.shape:", X_train.shape)

print("y_train.shape:", y_train.shape)

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(
        X_valid[col, :, 0],
        y_valid[col, 0],
        y_label=("$x(t)$" if col == 0 else None),
        legend=(col == 0),
    )
plt.show()


# ## Computing Some Baselines

# Naive predictions (just predict the last observed value):
y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()


# Linear predictions:


model = keras.models.Sequential(
    [keras.layers.Flatten(input_shape=[50, 1]), keras.layers.Dense(1)]
)

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))


model.evaluate(X_valid, y_valid)


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

# import libraries
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_functs import *

import matplotlib as mpl

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

####################################################################
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


# setting random seeds
np.random.seed(42)
tf.random.set_seed(42)

######################################################################
n_steps = 50
series = generate_time_series(10000, n_steps + 1)

###############################################################################
# test train val split
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
print(X_train.shape, y_train.shape)

###################################################################
#
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

y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

#####################################################################
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

#####################################################################

model = keras.models.Sequential(
    [
        # Flatten layer to flatten the input shape of [50, 1] to a 1D array of size 50
        keras.layers.Flatten(input_shape=[50, 1]),
        # Dense layer with 1 unit, representing a single output value
        keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluate the model on the validation set
model.evaluate(X_valid, y_valid)
###############################################################################

# Plot the learning curves
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# Make predictions on the validation set
y_pred = model.predict(X_valid)

# Plot the series
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

###############################################################################
# Using a Simpl RNN

# Define the model architecture
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(1, input_shape=[None, 1]),
    ],
)

# Define the optimizer with a specific learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.005)

# Compile the model
model.compile(loss="mse", optimizer=optimizer)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid),
)


model.evaluate(X_valid, y_valid)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

# Deep RNN


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1),
    ]
)

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

##############################################################################
### Make the second SimpleRNN layer return only the last output

# Define the model architecture
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluate the model on the validation set
model.evaluate(X_valid, y_valid)

# Plot the learning curves
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# Make predictions on the validation set
y_pred = model.predict(X_valid)

# Plot the series
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()
###############################################################################
# Forecasting Several Steps Ahead
np.random.seed(43)  # not 42, as it would give the first series in the train set

series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]


print("Y_pred.shape:", Y_pred.shape)


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "bo-", label="Actual")
    plt.plot(
        np.arange(n_steps, n_steps + ahead),
        Y_pred[0, :, 0],
        "rx-",
        label="Forecast",
        markersize=10,
    )
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)


plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()

# Now let's use this model to predict the next 10 values. We first need to regenerate the sequences with 9 more time steps

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

# Now let's predict the next 10 values one by one:
X = X_valid
for step_ahead in range(10):
    y_pred_one = model.predict(X)[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:, 0]

print("Y_pred.shape:", Y_pred.shape)

print("MSE:", np.mean(keras.metrics.mean_squared_error(Y_valid, Y_pred)))

# Let's compare this performance with some baselines: naive predictions
# and a simple linear model:


# take the last time step value, and repeat it 10 times
Y_naive_pred = np.tile(X_valid[:, -1], 10)
np.mean(keras.metrics.mean_squared_error(Y_valid, Y_naive_pred))

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential(
    [keras.layers.Flatten(input_shape=[50, 1]), keras.layers.Dense(10)]
)

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))

# Now let's create an RNN that predicts all 10 next values at once:

model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(10),
    ]
)

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]


plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()


#   create an RNN that predicts the next 10 steps at each time

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead : step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

X_train.shape, Y_train.shape


model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10)),
    ]
)


model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=[last_time_step_mse],
)
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]


plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()

###################################################################

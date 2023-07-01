# import libraries
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

###############################################################################
# setting random seeds
np.random.seed(42)
tf.random.set_seed(42)

###############################################################################
# custom functions


def generate_time_series(batch_size, n_steps):
    # Generate random values for frequencies and offsets
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)

    # Generate a time array from 0 to 1 with n_steps
    time = np.linspace(0, 1, n_steps)

    # Generate the first wave by applying a sinusoidal function to the time array
    # The frequency and offset values are used to modify the wave
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1

    # Generate the second wave by applying another sinusoidal function to the time array
    # Similar to the first wave, frequency and offset values are used to modify the wave
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2

    # Add random noise to the series by subtracting 0.5 from a random array and scaling it by 0.1
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise

    # Reshape the series to have an additional dimension using np.newaxis
    # and convert it to the float32 data type
    return series[..., np.newaxis].astype(np.float32)


def plot_series(
    series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True
):
    # Plot the series
    plt.plot(series, ".-")

    # Plot the target values if provided
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")

    # Plot the predicted values if provided
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")

    # Enable grid lines on the plot
    plt.grid(True)

    # Set the x-axis label if provided
    if x_label:
        plt.xlabel(x_label, fontsize=16)

    # Set the y-axis label if provided
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)

    # Add a horizontal line at y=0
    plt.hlines(0, 0, 100, linewidth=1)

    # Set the axis limits for the plot
    plt.axis([0, n_steps + 1, -1, 1])

    # Add legend if enabled and if there are target or predicted values
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")


def plot_learning_curves(loss, val_loss):
    """
    Plot the learning curves for training and validation loss.

    Args:
        loss (list): Training loss values.
        val_loss (list): Validation loss values.
    """
    # Plot the training loss curve
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")

    # Plot the validation loss curve
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")

    # Set the x-axis to display integers only
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # Set the axis limits for the plot
    plt.axis([1, 20, 0, 0.05])

    # Add a legend to the plot
    plt.legend(fontsize=14)

    # Set the x-axis label
    plt.xlabel("Epochs")

    # Set the y-axis label
    plt.ylabel("Loss")

    # Enable grid lines on the plot
    plt.grid(True)


###############################################################################
n_steps = 50
series = generate_time_series(10000, n_steps + 1)

###############################################################################
# test train val split
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
print(X_train.shape, y_train.shape)

###############################################################################
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

###############################################################################
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

###############################################################################

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

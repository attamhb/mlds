####################################################################
# custom functions

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


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
    series,
    y=None,
    y_pred=None,
    x_label="$t$",
    y_label="$x(t)$",
    legend=True,
    n_steps=50,
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

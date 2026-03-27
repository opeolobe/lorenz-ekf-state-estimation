#### Imports ######
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2




###### Plot State Estimates ##########
def plot_state_estimates(x_true, x_est, state_label=["x", "y", "z"]):
    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    ax = ax.flatten()

    # x plot
    ax[0].plot(x_true[:, 0], color='blue', label="Truth")
    ax[0].plot(x_est[:, 0], color="red", linestyle='--', label="ekf")
    ax[0].set_ylabel(state_label[0])
    ax[0].legend()

    # y plot
    ax[1].plot(x_true[:, 1], color='blue', label="Truth")
    ax[1].plot(x_est[:, 1], color="red", linestyle='--', label="ekf")
    ax[1].set_ylabel(state_label[1])
    ax[1].legend()

    # z plot
    ax[2].plot(x_true[:, 2], color='blue', label="Truth")
    ax[2].plot(x_est[:, 2], color="red", linestyle='--', label="ekf")
    ax[2].set_ylabel(state_label[2])
    ax[2].set_xlabel("Sample")
    ax[2].legend()

    fig.suptitle("Plot of the true states and estimated states (single trajectory)")
    fig.tight_layout()

    return fig, ax


def plot_nis_and_nees(nis_mean, nees_mean, T, N, num_outputs=2, num_states=3):
    # Trajectory length
    t = np.arange(T)

    m = num_outputs  # measurement dimension
    n = num_states   # state dimension

    # Confidence bounds for ensemble average
    alpha = 0.05
    lower_nis  = chi2.ppf(alpha/2, m*N) / N
    upper_nis  = chi2.ppf(1-alpha/2, m*N) / N

    lower_nees = chi2.ppf(alpha/2, n*N) / N
    upper_nees = chi2.ppf(1-alpha/2, n*N) / N

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top: NIS
    axes[0].plot(t, nis_mean, label="Mean NIS")
    axes[0].axhline(m, linestyle=":", color="green", label="Expected = 2")
    axes[0].axhline(lower_nis, linestyle="--", color="red", label="95% bounds")
    axes[0].axhline(upper_nis, linestyle="--", color="red")
    axes[0].set_ylabel("NIS")
    axes[0].set_title("Monte Carlo Consistency Test")
    axes[0].legend()

    # Bottom: NEES 
    axes[1].plot(t, nees_mean, label="Mean NEES")
    axes[1].axhline(n, linestyle=":", color="green", label="Expected = 3")
    axes[1].axhline(lower_nees, linestyle="--", color="red", label="95% bounds")
    axes[1].axhline(upper_nees, linestyle="--", color="red")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("NEES")
    axes[1].legend()

    plt.tight_layout()

    return fig, axes

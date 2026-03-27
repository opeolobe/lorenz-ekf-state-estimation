
"""
State estimation of a lorenz attractor model using EKF.
Lorenz has 3 states (x, y and z). Two of them (x and z) are measured as output corrupted with noise.

#*****************************************************************
# Date:         February, 2026
# Author:       Opeoluwa Adebayo, ooadebayo@mun.ca
# Institution:  Memorial University of Newfoundland, St. John's, NL
#*****************************************************************
"""


##*** Necessary Imports ***##
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from lorenz import simulate_sequence, ouputMatrix, lorenz_f, discrete_F, LorenzParameters, DataConfig
from ekf import ExtendedKalmanFilter
from plots import plot_state_estimates, plot_nis_and_nees


# Set random seed for reproducibility
np.random.seed(42)


##**** Monte Carlo Simulation of EKF for lorenz system ***##
def run_lorenz_ekf(cfg: DataConfig, p: LorenzParameters, N: int = 50):
    """State estimation for lorenz attractor across N trajectories."""

    dt, T, burn_in = cfg.dt, cfg.T, cfg.burn_in                         # Simulation parameters
    x_rmse_log, y_rmse_log, z_rmse_log = [], [], []                     # Logging rmse per state for each sequence
    nis_log = np.zeros((N, T))                                          # Logging of Normalized Innovation Squared
    nees_log = np.zeros((N, T))                                         # Logging of Normalized Estimation Error Squared
    first_run = True

    print("Estimating...")
    for seq in range(N):
        x_true, y_meas = simulate_sequence(cfg, p)                      # Simulate a long sequence
        x_est = np.zeros((cfg.T, x_true.shape[1]), dtype=np.float64)    # Estimated states
        x0 = x_true[0]                                                  # Initial states
        P0 = np.diag([0.01]*x_true.shape[1])                            # Initial state covariance
        Q = np.diag([0.2**2]*x_true.shape[1])                           # Process noise covariance (Tunable)
        R = np.diag([0.5**2]*y_meas.shape[1])                           # Measurement noise covariance (Tunable)
        H = ouputMatrix()                                               # Output matrix

        ekf = ExtendedKalmanFilter(x0=x0, P0=P0, Q=Q, R=R)              # Ekf object

        for k in range(T):
            ekf.predict(f_con=lorenz_f, F=discrete_F, dt=dt, args=(p,))
            x_hat, Pk, nis = ekf.update(y=y_meas[k], H=H)

            # Compute normalized estimation error squared (nees)
            error = x_true[k] - x_hat
            c, lower = cho_factor(Pk, check_finite=False)
            nees = float(error.T @ cho_solve((c, lower), error, check_finite=False))

            # Log variables
            x_est[k] = x_hat
            nis_log[seq, k] = nis
            nees_log[seq, k] = nees

        # Log variables
        error = x_true - x_est         
        rmse_per_state = np.sqrt(np.mean(error**2, axis=0))
        x_rmse, y_rmse, z_rmse = rmse_per_state
        x_rmse_log.append(x_rmse)
        y_rmse_log.append(y_rmse)
        z_rmse_log.append(z_rmse)

        # Show a sample plot for the state estimates
        if first_run:
            plot_state_estimates(x_true=x_true, x_est=x_est)
    
        # Disable first run
        first_run = False

    # RMSE Evaluation
    x_rmse_mean, x_rmse_std = np.array(x_rmse_log).mean(), np.array(x_rmse_log).std()
    y_rmse_mean, y_rmse_std = np.array(y_rmse_log).mean(), np.array(y_rmse_log).std()
    z_rmse_mean, z_rmse_std = np.array(z_rmse_log).mean(), np.array(z_rmse_log).std()

    print(f"Performance on {N} Sequence runs:")
    print("=" * 30)
    print(f"{'State':<6}{'RMSE':>15}")
    print(f"{'x':<10}{x_rmse_mean:>8.4f} ± {x_rmse_std:<7.4f}")
    print(f"{'y':<10}{y_rmse_mean:>8.4f} ± {y_rmse_std:<7.4f}")
    print(f"{'z':<10}{z_rmse_mean:>8.4f} ± {z_rmse_std:<7.4f}")
    print("=" * 30)

    # NIS and NEES Evaluation
    nis_mean_time  = np.mean(nis_log, axis=0)   
    nees_mean_time = np.mean(nees_log, axis=0)

    nis_ss  = np.mean(nis_mean_time[burn_in:])
    nees_ss = np.mean(nees_mean_time[burn_in:])

    print(f"\nStatistical Consistency")
    print("="*30)
    print(f"Global-SS NIS = {nis_ss:.4f}")
    print(f"Global-SS NEES = {nees_ss:.4f}")
    print("="*30)

    # Plot trajectory average NIS and NEES
    plot_nis_and_nees(nis_mean=nis_mean_time, nees_mean=nees_mean_time, T=T, N=N)



if __name__ == "__main__":
    p = LorenzParameters()
    cfg = DataConfig()
    N = 500

    run_lorenz_ekf(cfg, p, N)

    plt.show()
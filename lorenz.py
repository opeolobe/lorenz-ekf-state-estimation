"""
Helper functions to simulate lorenz attractor model for state estimation using EKF.
Lorenz model has 3 states (x, y and z). Two of the states (x and z) are measured as outputs
corrupted with measurement noise.

"""

##*** Necessary Libraries ***##
import numpy as np
from dataclasses import dataclass



##*** Helping Functions and Classes ***##
@dataclass
class LorenzParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


@dataclass
class DataConfig:
    dt: float = 0.01             # Sampling time
    T: int = 2000                # Sequence length
    burn_in: int = 200           # Burn-in period
    x0_range: float = 10.0       # range of initial condition
    pro_std: float = 0.2         # std of process noise
    meas_std_x: float = 0.5      # std of measurement noise for x-sensor
    meas_std_z: float = 0.5      # std of measurement noise for z-sensor



def lorenz_f(x: np.ndarray, p:LorenzParameters):
    """Continuous time dynamics for lorenz attractor: dxdt = f(x)."""
    X, Y, Z = x
    dx = p.sigma * (Y - X)
    dy = X * (p.rho - Z) - Y
    dz = X*Y - p.beta * Z
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_step(x: np.ndarray, dt: float, p:LorenzParameters):
    """Discretized dynamics for lorenz attractor using runge kutta."""
    k1 = lorenz_f(x, p)
    k2 = lorenz_f(x + 0.5 * dt * k1, p)
    k3 = lorenz_f(x + 0.5 * dt * k2, p)
    k4 = lorenz_f(x + dt * k3, p)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def jacobian_continuous(x: np.ndarray, p:LorenzParameters):
    """Continuous-time Jacobian of lorenz attractor. """
    sigma, rho, beta = p.sigma, p.rho, p.beta
    J = np.zeros((3, 3), dtype=np.float64)    # Jacobian initialization
    J[0, 0] = -sigma
    J[0, 1] = sigma
    J[0, 2] = 0.0
    J[1, 0] = rho - x[2]
    J[1, 1] = -1.0
    J[1, 2] = x[0]
    J[2, 0] = x[1]
    J[2, 1] = x[0]
    J[2, 2] = -beta
    return J

def discrete_F(x: np.ndarray, dt: float, p:LorenzParameters):
    """Discrete Jacobian of lorenz attractor using euler approximation."""
    J = jacobian_continuous(x, p)
    I = np.eye(3, dtype=np.float64)
    return I + dt * J



def ouputMatrix():
    """Linear output matrix for the lorenz system."""
    H = np.zeros((2, 3), dtype=np.float64)      # 2 outputs and 3 states
    H[0, 0] = 1.0                               # for x
    H[1, 2] = 1.0                               # for z
    return H



def simulate_sequence(cfg: DataConfig, p: LorenzParameters):
    """A long sequence simulation of the lorenz attractor."""
    dt, T = cfg.dt, cfg.T                                                                       # Simulation parameters
    x_std , z_std = cfg.meas_std_x, cfg.meas_std_z                                              # Measurement noise std
    w_std = cfg.pro_std                                                                         # Process noise std

    x = np.random.uniform(-cfg.x0_range, cfg.x0_range, size=(3,)).astype(np.float64)            # Initial condition for states

    x_true = np.zeros((T, 3), dtype=np.float64)                                                 # Ground truth states for evaluation
    y_meas = np.zeros((T, 2), dtype=np.float64)                                                 # Measured outputs

    for k in range(T):
        x = rk4_step(x, dt, p)
        x += np.random.normal(loc=0.0, scale=w_std, size=x.shape)                               # Process noise
        x_true[k] = x                                                                           # Store ground truth states

        y = np.array([x[0], x[2]], dtype=np.float64)                                            # Selected outputs
        y += np.array([np.random.randn()* x_std, np.random.randn() * z_std], dtype=np.float64)  # Measurement noise
        y_meas[k] = y

    return x_true, y_meas
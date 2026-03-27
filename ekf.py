"""
Generic Extended Kalman Filter using euler's approximation for state discretization
"""

##*** Necessary Libraries ***##
import numpy as np
from scipy.linalg import cho_factor, cho_solve


##*** Extended Kalman Filter ***###
class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R):
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1)   # Initial states (n,)
        self.P = np.asarray(P0, dtype=np.float64)               # Initial state covariance matrix
        self.Q = np.asarray(Q, dtype=np.float64)                # Process noise covariance
        self.R = np.asarray(R, dtype=np.float64)                # Measurement noise covariance
        self.I = np.eye(self.x.size, dtype=np.float64)

    def predict(self, f_con, F, dt, args=()):
        """
        f_con: continuous state transition function. EKF will 
        use euler approximation to discretize. This will create discretization mismatch.
        F: Discretized jacobian fucntion
        dt: Sampling time
        """
        x_prev = self.x                                         
        self.x = x_prev + dt * f_con(x_prev, *args)             # A priori states (Discretization mismatch)
        Fk = F(x_prev, dt, *args)                               # State transition Jacobian at the previous time step
        self.P = Fk @ self.P @ Fk.T + self.Q                    # A priori state error covariance
        self.P = 0.5 * (self.P + self.P.T)                      # Enforce symmetry

    def update(self, y, H):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        # H = np.astype(H, dtype=np.float64)

        y_hat = H @ self.x                                     # Predicted output or measurement (A priori)
        v = y - y_hat                                          # Innovation

        S = H @ self.P @ H.T + self.R                          # Innovation covariance
        S = 0.5 * (S + S.T)
    
        # Kalman gain using cholesky factorization (k = P_aprior @ H.T @ S^-1)
        PHt = self.P @ H.T                  
        c, lower = cho_factor(S, check_finite=False)
        K = cho_solve((c, lower), PHt.T, check_finite=False).T  

        # State update
        self.x = self.x + K @ v                                 # Aposteriori states

        # Joseph covariance update for state covariance (more stable)
        KH = K @ H                              
        A = self.I - KH
        self.P = A @ self.P @ A.T + K @ self.R @ K.T       
        self.P = 0.5 * (self.P + self.P.T)                      # Enforce symmetry

        # Normalized innovation squared (v.T @ S^-1 @ V)
        c, lower = cho_factor(S, check_finite=False)
        z = cho_solve((c, lower), v, check_finite=False)
        nis = float(v.T @ z)                                

        return self.x, self.P, nis

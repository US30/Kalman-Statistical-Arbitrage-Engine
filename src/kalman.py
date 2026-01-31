import numpy as np
import pandas as pd

class KalmanFilterReg:
    """
    A Kalman Filter implementation specifically for dynamic regression (Pairs Trading).
    
    Model:
    1. State Equation (Hidden):   Beta_t = Beta_{t-1} + w_t    (Beta follows a Random Walk)
    2. Observation Equation:      Y_t    = Beta_t * X_t + v_t  (Line equation with noise)
    
    Variables:
    - X: Independent Variable (Price of Stock B)
    - Y: Dependent Variable (Price of Stock A)
    - Delta: Process Noise Covariance (How much we allow Beta to vary)
    - R: Measurement Noise Covariance (How much noise is in the spread)
    """
    
    def __init__(self, delta=1e-5, R=1e-3):
        self.delta = delta  # Process noise (Q)
        self.R = R          # Measurement noise (R)
        
        # Initial State Estimates
        self.state_mean = 0  # Initial Beta
        self.state_cov = 1   # Initial Uncertainty (P)
        
    def process_data(self, x_series: pd.Series, y_series: pd.Series):
        """
        Runs the Recursive Filter over the dataset.
        
        Returns:
            pd.DataFrame containing:
            - 'beta': The dynamic hedge ratio
            - 'spread': The prediction error (Tradeable Signal)
            - 'var': The variance of the prediction error (for Z-Score normalization)
        """
        n = len(x_series)
        X = x_series.values
        Y = y_series.values
        
        # Output Containers
        beta_estimates = np.zeros(n)
        spread_errors = np.zeros(n)
        error_vars = np.zeros(n)
        
        # --- THE RECURSIVE LOOP ---
        for t in range(n):
            # 1. PREDICT STEP
            # Project state ahead (Random Walk: x_t = x_{t-1})
            pred_state = self.state_mean 
            # Project error covariance (P_t = P_{t-1} + Q)
            pred_cov = self.state_cov + self.delta
            
            # 2. UPDATE STEP
            # Observation Matrix H is simply the price of X at time t
            H = X[t]
            
            # Calculate Innovation (The "Spread")
            # This is the difference between Actual Y and Expected Y (based on prev Beta)
            y_pred = H * pred_state
            error = Y[t] - y_pred 
            
            # Innovation Covariance (S = H*P*H' + R)
            S = H * pred_cov * H + self.R
            
            # Kalman Gain (K = P*H' / S)
            # Determines how much we trust the new data vs our old model
            K = (pred_cov * H) / S
            
            # Update State Estimate (New Beta = Old Beta + Gain * Error)
            self.state_mean = pred_state + K * error
            
            # Update Covariance (Uncertainty)
            self.state_cov = pred_cov * (1 - K * H)
            
            # Store values
            beta_estimates[t] = self.state_mean
            spread_errors[t] = error
            error_vars[t] = S # We save 'S' to normalize the spread later
            
        return pd.DataFrame({
            'beta': beta_estimates,
            'spread': spread_errors,
            'spread_var': error_vars
        }, index=x_series.index)
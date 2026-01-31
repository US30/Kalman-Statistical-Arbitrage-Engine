import pandas as pd
import numpy as np

class StrategyAnalyzer:
    """
    Takes the output of the Kalman Filter and generates trading signals.
    """
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.0):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def generate_signals(self, kf_results: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Signal columns to the dataframe.
        
        Logic:
        - Z-Score = Spread / sqrt(Spread_Variance)
        - Long Signal (1): Z < -Threshold
        - Short Signal (-1): Z > Threshold
        - Exit (0): Z crosses 0
        """
        df = kf_results.copy()
        
        # 1. Calculate Z-Score (Dynamic Normalization)
        # We use sqrt(spread_var) because the Kalman Filter outputs Variance (S), 
        # but Z-score needs Standard Deviation.
        df['z_score'] = df['spread'] / np.sqrt(df['spread_var'])
        
        # 2. Vectorized Signal Generation
        df['long_entry'] = df['z_score'] < -self.entry_threshold
        df['short_entry'] = df['z_score'] > self.entry_threshold
        
        # Exits (Mean Reversion)
        # We exit longs when Z rises back above -0.5 (or 0)
        # We exit shorts when Z falls back below 0.5 (or 0)
        # For simplicity in this phase, we look for Z crossing Zero.
        df['exit_long'] = df['z_score'] >= -self.exit_threshold
        df['exit_short'] = df['z_score'] <= self.exit_threshold
        
        # 3. Create a Consolidated 'Position' Column (1, -1, 0)
        # This requires a loop because current position depends on previous state
        # (State Machine logic)
        positions = np.zeros(len(df))
        current_pos = 0 # 0=Flat, 1=Long, -1=Short
        
        z_scores = df['z_score'].values
        
        for t in range(len(df)):
            z = z_scores[t]
            
            if current_pos == 0:
                if z < -self.entry_threshold:
                    current_pos = 1  # Open Long
                elif z > self.entry_threshold:
                    current_pos = -1 # Open Short
            
            elif current_pos == 1: # We are Long
                if z >= -self.exit_threshold:
                    current_pos = 0 # Close Position
            
            elif current_pos == -1: # We are Short
                if z <= self.exit_threshold:
                    current_pos = 0 # Close Position
            
            positions[t] = current_pos
            
        df['position'] = positions
        
        return df
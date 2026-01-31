import pandas as pd
import numpy as np

class VectorizedBacktester:
    """
    Simulates the performance of the pair trading strategy.
    Assumes daily rebalancing of the hedge ratio (Beta).
    """
    
    def __init__(self, initial_capital=10000.0, transaction_cost_bps=0.0005):
        """
        Args:
            initial_capital: Starting cash.
            transaction_cost_bps: Basis points per trade (e.g., 5 bps = 0.0005).
        """
        self.initial_capital = initial_capital
        self.cost_bps = transaction_cost_bps
        
    def run(self, signals: pd.DataFrame, ticker_x: str, ticker_y: str):
        """
        Calculates PnL based on positions and price changes.
        """
        df = signals.copy()
        
        # 1. Calculate Daily Price Changes ($ per share)
        # We use diff() because we hold physical shares, not percent returns
        df['dY'] = df[ticker_y].diff() # Change in Stock A
        df['dX'] = df[ticker_x].diff() # Change in Stock B
        
        # 2. Strategy PnL Formula
        # We hold 1 share of Y and Short 'Beta' shares of X
        # PnL = Position_{t-1} * ( Change_Y - Beta_{t-1} * Change_X )
        df['strategy_pnl_daily'] = df['position'].shift(1) * (df['dY'] - df['beta'].shift(1) * df['dX'])
        
        # 3. Transaction Costs
        # We pay costs whenever the position CHANGES (Buy or Sell)
        # Cost estimate: |Change in Position| * Price * Cost_Bps
        # Assuming avg price of ~$50 for simplicity in this vectorization, or use actual price
        df['trades'] = df['position'].diff().abs()
        
        # Rough estimation of transaction value: Price_Y + Beta * Price_X
        # We approximate trade value as roughly $100 per unit spread for cost calc
        est_trade_value = (df[ticker_y] + df['beta'] * df[ticker_x]) 
        df['costs'] = df['trades'] * est_trade_value * self.cost_bps
        
        # Net PnL
        df['net_pnl'] = df['strategy_pnl_daily'] - df['costs'].fillna(0)
        
        # 4. Equity Curve
        # We assume we allocate capital to hold roughly 100 units of the spread
        # (Scaling factor to make the PnL meaningful relative to capital)
        # For M.Tech, we can just sum the PnL per 1 unit spread
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        
        # 5. Performance Metrics
        total_return = df['cumulative_pnl'].iloc[-1]
        sharpe = self._calculate_sharpe(df['net_pnl'])
        drawdown = self._calculate_drawdown(df['cumulative_pnl'])
        
        return {
            'equity_curve': df['cumulative_pnl'],
            'total_pnl': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'full_df': df
        }
    
    def _calculate_sharpe(self, daily_pnl_series):
        # Annualized Sharpe Ratio (assuming 252 trading days)
        if daily_pnl_series.std() == 0: return 0
        return np.sqrt(252) * (daily_pnl_series.mean() / daily_pnl_series.std())
    
    def _calculate_drawdown(self, equity_curve):
        # Rolling Max
        peak = equity_curve.cummax()
        drawdown = equity_curve - peak
        return drawdown.min()
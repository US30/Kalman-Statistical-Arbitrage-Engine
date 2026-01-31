import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import fetch_pair_data
from src.kalman import KalmanFilterReg
from src.strategy import StrategyAnalyzer
from src.backtester import VectorizedBacktester

def run_project():
    print("=== Phase 4: Full Backtest Execution ===")
    
    # 1. Config
    TICKER_X = 'PEP'
    TICKER_Y = 'KO'
    START = '2020-01-01'
    END = '2026-01-01'
    
    # 2. Pipeline Execution
    print(f"[1/4] Fetching Data ({TICKER_X} vs {TICKER_Y})...")
    df = fetch_pair_data(TICKER_X, TICKER_Y, START, END)
    if df.empty: return
    
    print("[2/4] Running Kalman Filter...")
    kf = KalmanFilterReg(delta=1e-4, R=1e-3)
    kf_results = kf.process_data(df[TICKER_X], df[TICKER_Y])
    
    print("[3/4] Generating Signals...")
    # Strict entry (2.0 std), Quick exit (mean reversion to 0)
    strat = StrategyAnalyzer(entry_threshold=2.0, exit_threshold=0.0)
    signals = strat.generate_signals(kf_results)
    
    # Merge price data back into signals for PnL calc
    signals[TICKER_X] = df[TICKER_X]
    signals[TICKER_Y] = df[TICKER_Y]
    
    print("[4/4] Calculating PnL...")
    # 5 bps transaction cost (0.05%)
    backtester = VectorizedBacktester(transaction_cost_bps=0.0005)
    metrics = backtester.run(signals, TICKER_X, TICKER_Y)
    
    # 3. Final Report
    print("\n" + "="*30)
    print("   STRATEGY PERFORMANCE REPORT   ")
    print("="*30)
    print(f"Total PnL (per 1 unit spread): ${metrics['total_pnl']:.2f}")
    print(f"Sharpe Ratio:                  {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:                  ${metrics['max_drawdown']:.2f}")
    print("="*30)
    
    # 4. Visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Cumulative PnL
    equity = metrics['equity_curve']
    axes[0].plot(equity.index, equity, color='green', linewidth=1.5)
    axes[0].fill_between(equity.index, equity, 0, where=(equity>0), color='green', alpha=0.1)
    axes[0].fill_between(equity.index, equity, 0, where=(equity<0), color='red', alpha=0.1)
    axes[0].set_title(f"Cumulative Profit/Loss ({TICKER_X}/{TICKER_Y})")
    axes[0].set_ylabel("PnL ($)")
    
    # Bottom Plot: The Spread and Entries
    spread = signals['z_score']
    axes[1].plot(spread.index, spread, label='Z-Score', color='gray', alpha=0.5)
    axes[1].axhline(2.0, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(-2.0, color='green', linestyle='--', alpha=0.5)
    
    # Highlight positions
    in_position = signals['position'] != 0
    axes[1].scatter(spread.index[in_position], spread.loc[in_position], 
                    color='blue', s=5, label='In Position', alpha=0.6)
    
    axes[1].set_title("Trading Signals (Z-Score)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_project()
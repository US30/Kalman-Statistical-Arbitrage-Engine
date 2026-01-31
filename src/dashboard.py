import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_loader import fetch_pair_data
from kalman import KalmanFilterReg
from strategy import StrategyAnalyzer
from backtester import VectorizedBacktester

st.set_page_config(page_title="Kalman Pairs Trader", layout="wide")

st.title("⚡ Kalman Filter Statistical Arbitrage Engine")
st.markdown("### Adaptive Pairs Trading Research Tool")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
ticker_x = st.sidebar.text_input("Benchmark Asset (X)", "PEP")
ticker_y = st.sidebar.text_input("Trading Asset (Y)", "KO")
delta = st.sidebar.select_slider("Kalman Delta (Process Noise)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
entry_z = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Fetching Data & Crunching Numbers..."):
        # 1. Pipeline Execution
        df = fetch_pair_data(ticker_x, ticker_y, "2020-01-01", "2026-01-01")
        
        if not df.empty:
            kf = KalmanFilterReg(delta=delta, R=1e-3)
            kf_results = kf.process_data(df[ticker_x], df[ticker_y])
            
            strat = StrategyAnalyzer(entry_threshold=entry_z, exit_threshold=0.0)
            signals = strat.generate_signals(kf_results)
            signals[ticker_x] = df[ticker_x]
            signals[ticker_y] = df[ticker_y]
            
            bt = VectorizedBacktester(transaction_cost_bps=0.0005)
            metrics = bt.run(signals, ticker_x, ticker_y)
            
            # --- Layout Results ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"${metrics['total_pnl']:.2f}")
            col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            col3.metric("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
            
            # --- Interactive Plots (Plotly) ---
            
            # 1. Equity Curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=metrics['equity_curve'].index, y=metrics['equity_curve'], 
                                            mode='lines', name='Equity', line=dict(color='#00ff00')))
            fig_equity.update_layout(title="Strategy Performance (PnL)", template="plotly_dark", height=400)
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 2. Dynamic Beta
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=kf_results.index, y=kf_results['beta'], 
                                          name='Kalman Beta', line=dict(color='cyan')))
            fig_beta.update_layout(title=f"Adaptive Hedge Ratio ({ticker_x}/{ticker_y})", template="plotly_dark", height=300)
            st.plotly_chart(fig_beta, use_container_width=True)
            
            # 3. Z-Score Signals
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=signals.index, y=signals['z_score'], name='Z-Score', line=dict(color='gray')))
            fig_z.add_hline(y=entry_z, line_dash="dash", line_color="red")
            fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="green")
            
            # Add Trades
            longs = signals[signals['position'] == 1]
            shorts = signals[signals['position'] == -1]
            fig_z.add_trace(go.Scatter(x=longs.index, y=longs['z_score'], mode='markers', name='Long', marker=dict(color='green', size=6)))
            fig_z.add_trace(go.Scatter(x=shorts.index, y=shorts['z_score'], mode='markers', name='Short', marker=dict(color='red', size=6)))
            
            fig_z.update_layout(title="Trading Signals", template="plotly_dark", height=300)
            st.plotly_chart(fig_z, use_container_width=True)
            
        else:
            st.error("No data found. Try different tickers.")
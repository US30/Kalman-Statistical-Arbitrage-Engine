import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import pandas as pd
import numpy as np

def check_cointegration(asset_a: pd.Series, asset_b: pd.Series):
    """
    Performs the Engle-Granger Cointegration Test.
    
    Step 1: OLS Regression (Asset_A ~ Asset_B) to find the spread.
    Step 2: ADF Unit Root Test on the residuals (spread).
    
    Returns:
        dict: Contains t-statistic, p-value, and critical values.
    """
    # Step 1: OLS Regression to find the static hedge ratio (beta)
    # We add a constant (intercept) to the independent variable
    X = sm.add_constant(asset_b) 
    model = sm.OLS(asset_a, X).fit()
    beta = model.params.iloc[1]
    
    # Calculate the spread (residuals)
    spread = asset_a - (beta * asset_b)
    
    # Step 2: Augmented Dickey-Fuller (ADF) Test on the spread
    # If p-value < 0.05, the spread is stationary -> Pairs are Cointegrated
    adf_result = ts.adfuller(spread)
    
    return {
        "ticker_a_price": asset_a.name,
        "ticker_b_price": asset_b.name,
        "hedge_ratio_ols": beta,
        "adf_stat": adf_result[0],
        "p_value": adf_result[1],
        "is_cointegrated": adf_result[1] < 0.05
    }
import pandas as pd
from curl_cffi import requests
from datetime import datetime
import time

def get_unix_timestamp(date_str: str) -> int:
    """Converts YYYY-MM-DD to Unix Timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(time.mktime(dt.timetuple()))

def fetch_single_ticker(ticker: str, start_str: str, end_str: str) -> pd.Series:
    """
    Fetches daily Adj Close data for a single ticker using curl_cffi 
    to bypass Yahoo Finance TLS fingerprinting.
    """
    period1 = get_unix_timestamp(start_str)
    period2 = get_unix_timestamp(end_str)
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }
    
    print(f"   -> Requesting {ticker} via curl_cffi (impersonating Chrome)...")
    
    try:
        # The Magic Line: impersonate="chrome" makes the server think we are a real browser
        response = requests.get(
            url, 
            params=params, 
            impersonate="chrome110",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse Yahoo's JSON structure
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        
        # We prefer Adjusted Close, but fallback to Close if needed
        if 'adjclose' in result['indicators']:
            prices = result['indicators']['adjclose'][0]['adjclose']
        else:
            prices = result['indicators']['quote'][0]['close']
            
        # Create Series
        ts_dates = pd.to_datetime(timestamps, unit='s')
        series = pd.Series(prices, index=ts_dates, name=ticker)
        
        # Remove timezone info to allow easy merging
        series.index = series.index.tz_localize(None)
        
        return series
        
    except Exception as e:
        print(f"❌ Failed to fetch {ticker}: {e}")
        return pd.Series(dtype=float)

def fetch_pair_data(ticker_a: str, ticker_b: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Orchestrator to fetch two assets and align them.
    """
    print(f"📥 Starting Custom Data Pipeline for {ticker_a} & {ticker_b}...")
    
    # Fetch individually
    s1 = fetch_single_ticker(ticker_a, start_date, end_date)
    s2 = fetch_single_ticker(ticker_b, start_date, end_date)
    
    if s1.empty or s2.empty:
        print("❌ Critical Error: One or both tickers returned no data.")
        return pd.DataFrame()
        
    # Align Data
    df = pd.concat([s1, s2], axis=1)
    
    # Drop NaNs (cleaning)
    original_len = len(df)
    df.dropna(inplace=True)
    
    print(f"✅ Data Aligned. {len(df)} rows ready (Dropped {original_len - len(df)} rows).")
    
    return df
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average with adjust=False."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculates MACD line, signal line, and histogram."""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "hist": histogram
    }

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """Calculates Bollinger Bands (upper, middle, lower, bandwidth)."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth
    }

def get_market_session(utc_dt: datetime) -> str:
    """
    Identifies market session based on UTC hour:
    London: 08:00 - 16:00 UTC
    New York: 13:00 - 21:00 UTC
    London/NY Overlap: 13:00 - 16:00 UTC (Highest liquidity & volume)
    Asian: 00:00 - 08:00 UTC
    Off hours / Low liquidity: 21:00 - 00:00 UTC
    """
    hour = utc_dt.hour
    if 13 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    elif 8 <= hour < 16:
        return "LONDON"
    elif 13 <= hour < 21:
        return "NEW_YORK"
    elif 0 <= hour < 8:
        return "ASIAN"
    else:
        return "SYDNEY_OFF_HOURS"

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all technical indicators on a DataFrame of OHLCV candles."""
    df = df.copy()
    if 'close' not in df or len(df) < 5:
        return df

    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['ema200'] = calculate_ema(df['close'], 200) if len(df) >= 200 else calculate_ema(df['close'], min(len(df), 50))
    df['rsi14'] = calculate_rsi(df['close'], 14)
    df['atr14'] = calculate_atr(df, 14)
    
    macd_res = calculate_macd(df['close'])
    df['macd_line'] = macd_res['macd']
    df['macd_signal'] = macd_res['signal']
    df['macd_hist'] = macd_res['hist']
    
    bb_res = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_res['upper']
    df['bb_middle'] = bb_res['middle']
    df['bb_lower'] = bb_res['lower']
    
    return df

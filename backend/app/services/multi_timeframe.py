import pandas as pd
from typing import Dict, Any, List
from app.schemas.analysis import MultiTimeframeSummary
from app.services.technical.indicators import compute_all_indicators
from app.services.technical.market_structure import analyze_market_structure

def evaluate_multi_timeframe(
    candles_4h: pd.DataFrame,
    candles_1h: pd.DataFrame,
    candles_15m: pd.DataFrame,
    candles_5m: pd.DataFrame
) -> MultiTimeframeSummary:
    """
    Evaluates multi-timeframe directional alignment:
    - 4H: Macro trend filter
    - 1H: Intermediate market structure
    - 15M: Primary setup timeframe
    - 5M: Precision entry trigger
    """
    def get_trend(df: pd.DataFrame) -> str:
        if len(df) < 5:
            return "UNCERTAIN"
        df_ind = compute_all_indicators(df)
        close = df_ind['close'].iloc[-1]
        ema20 = df_ind['ema20'].iloc[-1]
        ema50 = df_ind['ema50'].iloc[-1]
        
        if close > ema20 and ema20 > ema50:
            return "BULLISH"
        elif close < ema20 and ema20 < ema50:
            return "BEARISH"
        else:
            return "RANGING"

    t4 = get_trend(candles_4h)
    t1 = get_trend(candles_1h)
    t15 = get_trend(candles_15m)
    t5 = get_trend(candles_5m)
    
    # Calculate alignment score (0.0 to 1.0)
    # Check if timeframes align in the same direction
    bull_count = [t4, t1, t15, t5].count("BULLISH")
    bear_count = [t4, t1, t15, t5].count("BEARISH")
    
    max_align = max(bull_count, bear_count)
    alignment_score = max_align / 4.0
    
    return MultiTimeframeSummary(
        tf_4h_trend=t4,
        tf_1h_trend=t1,
        tf_15m_trend=t15,
        tf_5m_trend=t5,
        alignment_score=round(alignment_score, 2),
        confirmed=True
    )

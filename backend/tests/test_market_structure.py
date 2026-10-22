import pytest
import pandas as pd
from app.services.technical.market_structure import detect_swings, detect_bos_and_choch, cluster_support_resistance
from app.services.technical.price_action import detect_fair_value_gaps

def test_fair_value_gap_3_candle_rule():
    """
    Tests bullish and bearish FVG 3-candle rule.
    Bullish FVG: Candle 1 High < Candle 3 Low
    """
    candles = [
        {"timestamp": "2026-10-20T08:00:00Z", "open": 2720, "high": 2722, "low": 2719, "close": 2721}, # C1 High = 2722
        {"timestamp": "2026-10-20T08:15:00Z", "open": 2721, "high": 2732, "low": 2721, "close": 2731}, # C2 Imbalance
        {"timestamp": "2026-10-20T08:30:00Z", "open": 2731, "high": 2735, "low": 2726, "close": 2734}, # C3 Low = 2726
    ]
    df = pd.DataFrame(candles)
    fvgs = detect_fair_value_gaps(df, min_gap_points=0.5)
    
    assert len(fvgs) == 1
    assert fvgs[0].fvg_type == "BULLISH"
    assert fvgs[0].bottom == 2722.0 # C1 high
    assert fvgs[0].top == 2726.0    # C3 low

def test_swing_pivot_detection():
    """
    Tests non-repainting pivot detection with a lookback of 2.
    """
    candles = [
        {"high": 10, "low": 5, "close": 8},
        {"high": 12, "low": 6, "close": 11},
        {"high": 20, "low": 8, "close": 19}, # Peak
        {"high": 14, "low": 7, "close": 10},
        {"high": 11, "low": 4, "close": 6},  # Trough
        {"high": 15, "low": 8, "close": 14},
        {"high": 13, "low": 7, "close": 12},
    ]
    df = pd.DataFrame(candles)
    swings = detect_swings(df, lookback=2)
    
    highs = [s for s in swings if s.kind == "HIGH"]
    lows = [s for s in swings if s.kind == "LOW"]
    
    assert len(highs) >= 1
    assert highs[0].price == 20.0
    assert len(lows) >= 1
    assert lows[0].price == 4.0

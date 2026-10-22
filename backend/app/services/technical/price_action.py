import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from app.schemas.analysis import FVGZone, PriceActionPatterns, SwingPoint

def detect_fair_value_gaps(df: pd.DataFrame, min_gap_points: float = 0.5) -> List[FVGZone]:
    """
    Detects Fair Value Gaps (FVGs) using the canonical 3-candle definition:
    - Bullish FVG: Candle 1 High < Candle 3 Low (imbalance in Candle 2).
      FVG Zone = [Candle 1 High, Candle 3 Low]
    - Bearish FVG: Candle 1 Low > Candle 3 High (imbalance in Candle 2).
      FVG Zone = [Candle 3 High, Candle 1 Low]
    Checks whether subsequent candles have mitigated/filled the gap.
    """
    fvgs: List[FVGZone] = []
    n = len(df)
    if n < 3:
        return fvgs

    for i in range(2, n):
        c1_high = float(df['high'].iloc[i - 2])
        c1_low = float(df['low'].iloc[i - 2])
        c3_high = float(df['high'].iloc[i])
        c3_low = float(df['low'].iloc[i])
        timestamp_str = str(df['timestamp'].iloc[i - 1]) if 'timestamp' in df else str(i - 1)
        
        # Bullish FVG: c3_low > c1_high
        if c3_low - c1_high >= min_gap_points:
            # Check mitigation by subsequent candles i+1 to end
            mitigated = False
            for k in range(i + 1, n):
                if float(df['low'].iloc[k]) <= c1_high:
                    mitigated = True
                    break
                    
            fvgs.append(FVGZone(
                fvg_type="BULLISH",
                top=round(c3_low, 2),
                bottom=round(c1_high, 2),
                mid=round((c3_low + c1_high) / 2.0, 2),
                candle_index=i - 1,
                candle_timestamp=timestamp_str,
                mitigated=mitigated
            ))
            
        # Bearish FVG: c1_low > c3_high
        elif c1_low - c3_high >= min_gap_points:
            mitigated = False
            for k in range(i + 1, n):
                if float(df['high'].iloc[k]) >= c1_low:
                    mitigated = True
                    break
                    
            fvgs.append(FVGZone(
                fvg_type="BEARISH",
                top=round(c1_low, 2),
                bottom=round(c3_high, 2),
                mid=round((c1_low + c3_high) / 2.0, 2),
                candle_index=i - 1,
                candle_timestamp=timestamp_str,
                mitigated=mitigated
            ))
            
    return fvgs

def detect_equal_highs_lows(swings: List[SwingPoint], tolerance_pct: float = 0.0006) -> Dict[str, List[float]]:
    """Identifies Equal Highs (EQH) and Equal Lows (EQL) representing liquidity pools."""
    highs = [s.price for s in swings if s.kind == "HIGH"]
    lows = [s.price for s in swings if s.kind == "LOW"]
    
    eqh = []
    for i in range(len(highs) - 1):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] <= tolerance_pct:
                eqh.append(round((highs[i] + highs[j]) / 2.0, 2))
                
    eql = []
    for i in range(len(lows) - 1):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] <= tolerance_pct:
                eql.append(round((lows[i] + lows[j]) / 2.0, 2))
                
    return {"equal_highs": list(set(eqh)), "equal_lows": list(set(eql))}

def detect_liquidity_sweeps(df: pd.DataFrame, swings: List[SwingPoint]) -> List[Dict[str, Any]]:
    """
    Detects liquidity sweeps / stop hunts:
    - Bullish sweep: Price trades below confirmed swing low, but candle closes back inside/above.
    - Bearish sweep: Price trades above confirmed swing high, but candle closes back inside/below.
    """
    sweeps = []
    if len(df) < 5 or not swings:
        return sweeps

    recent_highs = [s.price for s in swings if s.kind == "HIGH"]
    recent_lows = [s.price for s in swings if s.kind == "LOW"]
    
    if not recent_highs or not recent_lows:
        return sweeps
        
    last_high = recent_highs[-1]
    last_low = recent_lows[-1]
    
    # Inspect the most recent 3 candles
    for idx in range(-3, 0):
        c_high = float(df['high'].iloc[idx])
        c_low = float(df['low'].iloc[idx])
        c_close = float(df['close'].iloc[idx])
        c_open = float(df['open'].iloc[idx])
        ts = str(df['timestamp'].iloc[idx]) if 'timestamp' in df else str(idx)
        
        # Bearish sweep of swing high
        if c_high > last_high and c_close < last_high:
            sweeps.append({
                "type": "BEARISH_SWEEP",
                "swept_level": last_high,
                "high": c_high,
                "close": c_close,
                "timestamp": ts,
                "note": f"Price swept swing high at {last_high:.2f} and closed back below at {c_close:.2f} (Bearish Liquidity Grab)"
            })
            
        # Bullish sweep of swing low
        if c_low < last_low and c_close > last_low:
            sweeps.append({
                "type": "BULLISH_SWEEP",
                "swept_level": last_low,
                "low": c_low,
                "close": c_close,
                "timestamp": ts,
                "note": f"Price swept swing low at {last_low:.2f} and closed back above at {c_close:.2f} (Bullish Liquidity Grab)"
            })
            
    return sweeps

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Detects pin bar rejections and engulfing patterns."""
    rejections = []
    engulfings = []
    n = len(df)
    if n < 2:
        return {"rejections": rejections, "engulfings": engulfings}

    # Inspect the last 3 candles
    for i in range(-3, 0):
        o = float(df['open'].iloc[i])
        h = float(df['high'].iloc[i])
        l = float(df['low'].iloc[i])
        c = float(df['close'].iloc[i])
        body = abs(c - o)
        total_range = h - l
        ts = str(df['timestamp'].iloc[i]) if 'timestamp' in df else str(i)
        
        if total_range > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            
            # Bullish Pin Bar / Hammer
            if lower_wick / total_range >= 0.55 and body / total_range <= 0.35:
                rejections.append({
                    "pattern": "BULLISH_PIN_BAR",
                    "price": c,
                    "timestamp": ts,
                    "rejection_level": l
                })
            # Bearish Pin Bar / Shooting Star
            elif upper_wick / total_range >= 0.55 and body / total_range <= 0.35:
                rejections.append({
                    "pattern": "BEARISH_PIN_BAR",
                    "price": c,
                    "timestamp": ts,
                    "rejection_level": h
                })
                
        # Engulfing (compare with previous candle)
        prev_o = float(df['open'].iloc[i - 1])
        prev_c = float(df['close'].iloc[i - 1])
        
        # Bullish Engulfing
        if prev_c < prev_o and c > o and c >= prev_o and o <= prev_c:
            engulfings.append({
                "pattern": "BULLISH_ENGULFING",
                "timestamp": ts,
                "price": c
            })
        # Bearish Engulfing
        elif prev_c > prev_o and c < o and c <= prev_o and o >= prev_c:
            engulfings.append({
                "pattern": "BEARISH_ENGULFING",
                "timestamp": ts,
                "price": c
            })
            
    return {"rejections": rejections, "engulfings": engulfings}

def analyze_price_action(df: pd.DataFrame, swings: List[SwingPoint]) -> PriceActionPatterns:
    """Full price action analysis."""
    fvgs = detect_fair_value_gaps(df)
    eq_levels = detect_equal_highs_lows(swings)
    sweeps = detect_liquidity_sweeps(df, swings)
    patterns = detect_candlestick_patterns(df)
    
    # Return unmitigated FVGs or recent 5
    active_fvgs = [f for f in fvgs if not f.mitigated][-5:]
    if not active_fvgs and fvgs:
        active_fvgs = fvgs[-5:]
        
    return PriceActionPatterns(
        equal_highs=eq_levels["equal_highs"],
        equal_lows=eq_levels["equal_lows"],
        liquidity_sweeps=sweeps,
        fair_value_gaps=active_fvgs,
        rejection_candles=patterns["rejections"],
        engulfing_patterns=patterns["engulfings"]
    )

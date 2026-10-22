import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from app.schemas.analysis import SwingPoint, SRZone, MarketStructureSummary

def detect_swings(df: pd.DataFrame, lookback: int = 4) -> List[SwingPoint]:
    """
    Detects confirmed swing highs and lows using an n-bar lookback window.
    Strictly non-repainting: A pivot at bar i requires lookback bars to the right,
    so it is only confirmed at bar i + lookback.
    """
    swings: List[SwingPoint] = []
    n = len(df)
    if n < lookback * 2 + 1:
        return swings

    for i in range(lookback, n - lookback):
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        
        # Check swing high
        is_high = True
        for offset in range(-lookback, lookback + 1):
            if offset != 0 and df['high'].iloc[i + offset] >= curr_high:
                is_high = False
                break
        
        # Check swing low
        is_low = True
        for offset in range(-lookback, lookback + 1):
            if offset != 0 and df['low'].iloc[i + offset] <= curr_low:
                is_low = False
                break
                
        timestamp_str = str(df['timestamp'].iloc[i]) if 'timestamp' in df else str(i)
        
        if is_high:
            swings.append(SwingPoint(
                price=float(curr_high),
                index=i,
                timestamp=timestamp_str,
                kind="HIGH"
            ))
        elif is_low:
            swings.append(SwingPoint(
                price=float(curr_low),
                index=i,
                timestamp=timestamp_str,
                kind="LOW"
            ))
            
    # Sort swings chronologically by index
    swings.sort(key=lambda s: s.index)
    
    # Classify HH, HL, LH, LL
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    
    for s in swings:
        if s.kind == "HIGH":
            if last_high is not None:
                s.label = "HH" if s.price > last_high else "LH"
            else:
                s.label = "HIGH"
            last_high = s.price
        elif s.kind == "LOW":
            if last_low is not None:
                s.label = "HL" if s.price > last_low else "LL"
            else:
                s.label = "LOW"
            last_low = s.price
            
    return swings

def detect_bos_and_choch(df: pd.DataFrame, swings: List[SwingPoint]) -> Dict[str, Any]:
    """
    Identifies Break of Structure (BOS) and Change of Character (CHoCH):
    - BOS Bullish: Price closes above previous swing high during an uptrend (continuation).
    - BOS Bearish: Price closes below previous swing low during a downtrend (continuation).
    - CHoCH Bullish: Downtrend (LH/LL) broken by price closing above last LH (early reversal).
    - CHoCH Bearish: Uptrend (HH/HL) broken by price closing below last HL (early reversal).
    """
    if len(swings) < 3 or len(df) == 0:
        return {
            "bos": False, "bos_dir": None,
            "choch": False, "choch_dir": None,
            "trend": "UNCERTAIN"
        }
        
    recent_highs = [s for s in swings if s.kind == "HIGH"]
    recent_lows = [s for s in swings if s.kind == "LOW"]
    
    if not recent_highs or not recent_lows:
        return {"bos": False, "bos_dir": None, "choch": False, "choch_dir": None, "trend": "UNCERTAIN"}
        
    last_high = recent_highs[-1].price
    last_low = recent_lows[-1].price
    prev_high = recent_highs[-2].price if len(recent_highs) >= 2 else last_high
    prev_low = recent_lows[-2].price if len(recent_lows) >= 2 else last_low
    
    current_close = float(df['close'].iloc[-1])
    
    # Determine base trend from recent swings
    is_uptrend = (last_high >= prev_high) and (last_low >= prev_low)
    is_downtrend = (last_high <= prev_high) and (last_low <= prev_low)
    
    bos = False
    bos_dir = None
    choch = False
    choch_dir = None
    
    if current_close > last_high:
        if is_uptrend:
            bos = True
            bos_dir = "BULLISH"
        else:
            choch = True
            choch_dir = "BULLISH"
    elif current_close < last_low:
        if is_downtrend:
            bos = True
            bos_dir = "BEARISH"
        else:
            choch = True
            choch_dir = "BEARISH"
            
    trend = "BULLISH" if is_uptrend else "BEARISH" if is_downtrend else "RANGING"
    
    return {
        "bos": bos,
        "bos_dir": bos_dir,
        "choch": choch,
        "choch_dir": choch_dir,
        "trend": trend,
        "last_high": last_high,
        "last_low": last_low
    }

def cluster_support_resistance(swings: List[SwingPoint], tolerance_pct: float = 0.002) -> List[SRZone]:
    """
    Clusters swing highs and lows into defined Support and Resistance zones.
    """
    if not swings:
        return []
        
    prices = [s.price for s in swings]
    zones: List[SRZone] = []
    
    # Separate highs (potential resistance) and lows (potential support)
    highs = [s.price for s in swings if s.kind == "HIGH"]
    lows = [s.price for s in swings if s.kind == "LOW"]
    
    def form_clusters(price_list: List[float], zone_type: str) -> List[SRZone]:
        clusters: List[SRZone] = []
        visited = set()
        for i, p in enumerate(price_list):
            if i in visited:
                continue
            group = [p]
            visited.add(i)
            for j, p2 in enumerate(price_list):
                if j not in visited and abs(p - p2) / p <= tolerance_pct:
                    group.append(p2)
                    visited.add(j)
            
            touches = len(group)
            min_l = min(group)
            max_l = max(group)
            mid = sum(group) / touches
            
            # Add breathing room if single touch
            if min_l == max_l:
                min_l = mid * (1 - tolerance_pct / 2)
                max_l = mid * (1 + tolerance_pct / 2)
                
            strength = "STRONG" if touches >= 3 else "MODERATE" if touches == 2 else "WEAK"
            clusters.append(SRZone(
                zone_type=zone_type,
                level_min=round(min_l, 2),
                level_max=round(max_l, 2),
                mid_price=round(mid, 2),
                touches=touches,
                strength=strength
            ))
        return clusters

    res_zones = form_clusters(highs, "RESISTANCE")
    sup_zones = form_clusters(lows, "SUPPORT")
    
    all_zones = res_zones + sup_zones
    # Sort zones by mid_price descending
    all_zones.sort(key=lambda z: z.mid_price, reverse=True)
    return all_zones

def analyze_market_structure(df: pd.DataFrame, lookback: int = 4) -> MarketStructureSummary:
    """Complete market structure analysis on candle dataframe."""
    swings = detect_swings(df, lookback)
    bos_choch = detect_bos_and_choch(df, swings)
    sr_zones = cluster_support_resistance(swings)
    
    last_high = bos_choch.get("last_high")
    last_low = bos_choch.get("last_low")
    
    return MarketStructureSummary(
        trend=bos_choch.get("trend", "UNCERTAIN"),
        last_swing_high=last_high,
        last_swing_low=last_low,
        swing_points=swings[-12:], # Return last 12 swings for chart display
        bos_detected=bos_choch.get("bos", False),
        bos_direction=bos_choch.get("bos_dir"),
        choch_detected=bos_choch.get("choch", False),
        choch_direction=bos_choch.get("choch_dir"),
        sr_zones=sr_zones[:8], # Top 8 most relevant zones
        market_regime="TRENDING" if bos_choch.get("trend") in ["BULLISH", "BEARISH"] else "RANGING"
    )

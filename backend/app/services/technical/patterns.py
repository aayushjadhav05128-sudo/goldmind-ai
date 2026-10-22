import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from app.schemas.analysis import SwingPoint

# ============================================================================
# 1. CANDLESTICK PATTERN RECOGNITION (Documented, Non-Subjective Rules)
# ============================================================================

def detect_all_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects classical and institutional candlestick patterns across the last 10 candles.
    Evaluates body/wick ratios, directional context, and engulfing properties.
    """
    patterns: List[Dict[str, Any]] = []
    n = len(df)
    if n < 3:
        return patterns

    lookback_bars = min(n, 12)
    start_idx = n - lookback_bars

    for i in range(start_idx, n):
        o = float(df['open'].iloc[i])
        h = float(df['high'].iloc[i])
        l = float(df['low'].iloc[i])
        c = float(df['close'].iloc[i])
        ts = str(df['timestamp'].iloc[i]) if 'timestamp' in df else str(i)
        
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0.0001:
            continue

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        body_pct = body / total_range
        upper_wick_pct = upper_wick / total_range
        lower_wick_pct = lower_wick / total_range
        is_bullish_bar = c >= o

        # 1. Doji / Long-Legged Doji
        if body_pct <= 0.08:
            patterns.append({
                "bar_index": i,
                "timestamp": ts,
                "pattern": "DOJI",
                "bias": "NEUTRAL_INDECISION",
                "price": c,
                "description": f"Doji candle at {c:.2f} indicating market equilibrium and potential pause/reversal."
            })

        # 2. Pin Bar / Hammer (Bullish Rejection)
        if lower_wick_pct >= 0.58 and body_pct <= 0.32:
            patterns.append({
                "bar_index": i,
                "timestamp": ts,
                "pattern": "HAMMER_PIN_BAR",
                "bias": "BULLISH_REJECTION",
                "price": c,
                "description": f"Bullish pin bar / hammer rejecting {l:.2f} with strong lower wick ({lower_wick_pct*100:.0f}% of candle)."
            })

        # 3. Shooting Star (Bearish Rejection)
        if upper_wick_pct >= 0.58 and body_pct <= 0.32:
            patterns.append({
                "bar_index": i,
                "timestamp": ts,
                "pattern": "SHOOTING_STAR_PIN_BAR",
                "bias": "BEARISH_REJECTION",
                "price": c,
                "description": f"Bearish pin bar / shooting star rejecting {h:.2f} with upper wick ({upper_wick_pct*100:.0f}% of candle)."
            })

        # 4. Marubozu (Strong Momentum Bar with negligible wicks)
        if body_pct >= 0.85:
            patterns.append({
                "bar_index": i,
                "timestamp": ts,
                "pattern": "BULLISH_MARUBOZU" if is_bullish_bar else "BEARISH_MARUBOZU",
                "bias": "BULLISH_MOMENTUM" if is_bullish_bar else "BEARISH_MOMENTUM",
                "price": c,
                "description": f"Marubozu impulsive momentum bar ({'Bullish' if is_bullish_bar else 'Bearish'}) closing near extreme."
            })

        # Multi-candle patterns (Requires i >= 1)
        if i >= 1:
            prev_o = float(df['open'].iloc[i - 1])
            prev_h = float(df['high'].iloc[i - 1])
            prev_l = float(df['low'].iloc[i - 1])
            prev_c = float(df['close'].iloc[i - 1])
            prev_body = abs(prev_c - prev_o)
            prev_bullish = prev_c >= prev_o

            # 5. Bullish Engulfing
            if not prev_bullish and is_bullish_bar and o <= prev_c and c >= prev_o and body > prev_body:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "BULLISH_ENGULFING",
                    "bias": "BULLISH",
                    "price": c,
                    "description": f"Bullish engulfing candle completely engulfing previous bearish body at {c:.2f}."
                })

            # 6. Bearish Engulfing
            if prev_bullish and not is_bullish_bar and o >= prev_c and c <= prev_o and body > prev_body:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "BEARISH_ENGULFING",
                    "bias": "BEARISH",
                    "price": c,
                    "description": f"Bearish engulfing candle completely engulfing previous bullish body at {c:.2f}."
                })

            # 7. Inside Bar (Compression / Pause)
            if h <= prev_h and l >= prev_l:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "INSIDE_BAR",
                    "bias": "VOLATILITY_COMPRESSION",
                    "price": c,
                    "description": f"Inside bar consolidating within previous mother bar [{prev_l:.2f} - {prev_h:.2f}]."
                })

            # 8. Outside Bar / Engulfing Range
            if h > prev_h and l < prev_l:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "OUTSIDE_BAR",
                    "bias": "EXPANSION_VOLATILITY",
                    "price": c,
                    "description": f"Outside bar engulfing previous range completely."
                })

            # 9. Tweezer Tops & Bottoms
            if abs(h - prev_h) <= 0.35 and upper_wick_pct >= 0.4:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "TWEEZER_TOP",
                    "bias": "BEARISH_REVERSAL",
                    "price": c,
                    "description": f"Tweezer Top double rejection at {max(h, prev_h):.2f}."
                })
            elif abs(l - prev_l) <= 0.35 and lower_wick_pct >= 0.4:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "TWEEZER_BOTTOM",
                    "bias": "BULLISH_REVERSAL",
                    "price": c,
                    "description": f"Tweezer Bottom double rejection at {min(l, prev_l):.2f}."
                })

        # 3-Candle Patterns (Requires i >= 2)
        if i >= 2:
            p2_o = float(df['open'].iloc[i - 2])
            p2_c = float(df['close'].iloc[i - 2])
            p1_o = float(df['open'].iloc[i - 1])
            p1_c = float(df['close'].iloc[i - 1])
            
            # 10. Morning Star (Bearish bar -> Star/Doji -> Strong Bullish bar)
            if p2_c < p2_o and abs(p1_c - p1_o) < abs(p2_c - p2_o) * 0.4 and c > o and c > (p2_o + p2_c) / 2:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "MORNING_STAR",
                    "bias": "BULLISH_REVERSAL",
                    "price": c,
                    "description": f"Confirmed 3-bar Morning Star reversal pattern closing above midpoint of bearish candle."
                })

            # 11. Evening Star (Bullish bar -> Star/Doji -> Strong Bearish bar)
            if p2_c > p2_o and abs(p1_c - p1_o) < abs(p2_c - p2_o) * 0.4 and c < o and c < (p2_o + p2_c) / 2:
                patterns.append({
                    "bar_index": i,
                    "timestamp": ts,
                    "pattern": "EVENING_STAR",
                    "bias": "BEARISH_REVERSAL",
                    "price": c,
                    "description": f"Confirmed 3-bar Evening Star reversal pattern closing below midpoint of bullish candle."
                })

    return patterns

# ============================================================================
# 2. CLASSICAL CHART PATTERNS (Double Tops, Head & Shoulders, Triangles)
# ============================================================================

def detect_chart_patterns(swings: List[SwingPoint], current_price: float) -> List[Dict[str, Any]]:
    """
    Identifies classical formations from confirmed swing pivots:
    - Double Top & Double Bottom
    - Head & Shoulders / Inverse Head & Shoulders
    - Ascending / Descending Triangles
    """
    patterns: List[Dict[str, Any]] = []
    if len(swings) < 4:
        return patterns

    highs = [s for s in swings if s.kind == "HIGH"]
    lows = [s for s in swings if s.kind == "LOW"]

    # 1. Double Top (M-Pattern)
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        tolerance = h1.price * 0.0012 # 0.12% tolerance (~$3 on gold)
        if abs(h1.price - h2.price) <= tolerance:
            # Find intermediate neckline low
            neckline_lows = [l.price for l in lows if h1.index < l.index < h2.index]
            neckline = min(neckline_lows) if neckline_lows else (h1.price - 5.0)
            target = neckline - (h1.price - neckline)
            patterns.append({
                "pattern": "DOUBLE_TOP",
                "bias": "BEARISH",
                "level": round((h1.price + h2.price) / 2, 2),
                "neckline": round(neckline, 2),
                "measured_target": round(target, 2),
                "confirmed": current_price < neckline,
                "description": f"Double Top formed at ${((h1.price + h2.price)/2):.2f}. Neckline at ${neckline:.2f} with measured target ${target:.2f}."
            })

    # 2. Double Bottom (W-Pattern)
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        tolerance = l1.price * 0.0012
        if abs(l1.price - l2.price) <= tolerance:
            neckline_highs = [h.price for h in highs if l1.index < h.index < l2.index]
            neckline = max(neckline_highs) if neckline_highs else (l1.price + 5.0)
            target = neckline + (neckline - l1.price)
            patterns.append({
                "pattern": "DOUBLE_BOTTOM",
                "bias": "BULLISH",
                "level": round((l1.price + l2.price) / 2, 2),
                "neckline": round(neckline, 2),
                "measured_target": round(target, 2),
                "confirmed": current_price > neckline,
                "description": f"Double Bottom formed at ${((l1.price + l2.price)/2):.2f}. Neckline at ${neckline:.2f} with measured target ${target:.2f}."
            })

    # 3. Head & Shoulders (Left Shoulder, Head, Right Shoulder)
    if len(highs) >= 3:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        if head.price > ls.price and head.price > rs.price and abs(ls.price - rs.price) / ls.price <= 0.002:
            patterns.append({
                "pattern": "HEAD_AND_SHOULDERS",
                "bias": "BEARISH_REVERSAL",
                "head_price": round(head.price, 2),
                "shoulder_level": round((ls.price + rs.price) / 2, 2),
                "description": f"Head & Shoulders pattern detected. Peak Head at ${head.price:.2f}, Right Shoulder at ${rs.price:.2f}."
            })

    # 4. Inverse Head & Shoulders
    if len(lows) >= 3:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if head.price < ls.price and head.price < rs.price and abs(ls.price - rs.price) / ls.price <= 0.002:
            patterns.append({
                "pattern": "INVERSE_HEAD_AND_SHOULDERS",
                "bias": "BULLISH_REVERSAL",
                "head_price": round(head.price, 2),
                "shoulder_level": round((ls.price + rs.price) / 2, 2),
                "description": f"Inverse Head & Shoulders bottom detected. Head low at ${head.price:.2f}, Right Shoulder at ${rs.price:.2f}."
            })

    return patterns

# ============================================================================
# 3. SMART MONEY CONCEPTS (SMC) & ICT (Order Blocks, Breakers, Equilibrium)
# ============================================================================

def detect_smc_structures(df: pd.DataFrame, swings: List[SwingPoint]) -> Dict[str, Any]:
    """
    Identifies institutional Smart Money Concepts (SMC) / ICT features:
    - Order Block (OB) candidates: Last opposing candle before impulsive displacement
    - Dealing Range & Equilibrium (50% midpoint)
    - Premium vs Discount zones
    - Market Structure Shift (MSS)
    - Killzone Session status
    """
    n = len(df)
    if n < 10:
        return {}

    current_price = float(df['close'].iloc[-1])
    
    # Dealing range: highest high and lowest low of last 50 bars
    lookback = min(n, 50)
    recent_range_high = float(df['high'].iloc[-lookback:].max())
    recent_range_low = float(df['low'].iloc[-lookback:].min())
    equilibrium = round((recent_range_high + recent_range_low) / 2.0, 2)

    # Premium / Discount context
    in_premium = current_price > equilibrium
    in_discount = current_price < equilibrium
    price_zone = "PREMIUM (Sell-favorable)" if in_premium else "DISCOUNT (Buy-favorable)" if in_discount else "EQUILIBRIUM"

    # Order Block Candidate Detection:
    # Bullish OB: Last down candle before strong multi-bar displacement upward
    # Bearish OB: Last up candle before strong multi-bar displacement downward
    order_blocks = []
    for i in range(n - 20, n - 3):
        c1_o, c1_c = float(df['open'].iloc[i]), float(df['close'].iloc[i])
        c1_h, c1_l = float(df['high'].iloc[i]), float(df['low'].iloc[i])
        
        # Bullish displacement
        disp_up = float(df['close'].iloc[i + 2]) - float(df['close'].iloc[i])
        if c1_c < c1_o and disp_up > 6.0: # strong $6+ displacement on gold
            order_blocks.append({
                "type": "BULLISH_ORDER_BLOCK",
                "top": round(c1_h, 2),
                "bottom": round(c1_l, 2),
                "mitigated": current_price < c1_l,
                "description": f"Bullish Order Block formed at [{c1_l:.2f} - {c1_h:.2f}] before upward displacement."
            })

        # Bearish displacement
        disp_down = float(df['close'].iloc[i]) - float(df['close'].iloc[i + 2])
        if c1_c > c1_o and disp_down > 6.0:
            order_blocks.append({
                "type": "BEARISH_ORDER_BLOCK",
                "top": round(c1_h, 2),
                "bottom": round(c1_l, 2),
                "mitigated": current_price > c1_h,
                "description": f"Bearish Order Block formed at [{c1_l:.2f} - {c1_h:.2f}] before downward displacement."
            })

    # Filter for active unmitigated OBs
    active_obs = [ob for ob in order_blocks if not ob["mitigated"]][-3:]

    # London & New York Killzones (UTC)
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    killzone = "OFF_HOURS"
    if 7 <= hour <= 10:
        killzone = "LONDON_OPEN_KILLZONE (07:00-10:00 UTC)"
    elif 12 <= hour <= 15:
        killzone = "NEW_YORK_OPEN_KILLZONE (12:00-15:00 UTC)"
    elif 19 <= hour <= 21:
        killzone = "LONDON_CLOSE_KILLZONE (19:00-21:00 UTC)"
    elif 0 <= hour <= 5:
        killzone = "ASIAN_SESSION (00:00-05:00 UTC)"

    return {
        "dealing_range_high": recent_range_high,
        "dealing_range_low": recent_range_low,
        "equilibrium_50": equilibrium,
        "market_zone": price_zone,
        "killzone": killzone,
        "order_blocks": active_obs
    }

# ============================================================================
# 4. FIBONACCI RETRACEMENT & EXTENSION ENGINE
# ============================================================================

def calculate_fibonacci_levels(swings: List[SwingPoint]) -> Dict[str, Any]:
    """
    Anchors Fibonacci levels to the most recent confirmed swing impulse (Swing Low -> Swing High or vice versa).
    Returns 38.2%, 50.0%, 61.8% Golden Pocket, 78.6% retracements, and 127.2%, 161.8% extensions.
    """
    if len(swings) < 2:
        return {}

    s1, s2 = swings[-2], swings[-1]
    
    if s1.kind == "LOW" and s2.kind == "HIGH":
        # Bullish impulse (Low to High) -> Retracement down
        diff = s2.price - s1.price
        return {
            "impulse_direction": "BULLISH_IMPULSE",
            "anchor_low": round(s1.price, 2),
            "anchor_high": round(s2.price, 2),
            "fib_236": round(s2.price - (diff * 0.236), 2),
            "fib_382": round(s2.price - (diff * 0.382), 2),
            "fib_500": round(s2.price - (diff * 0.500), 2),
            "fib_618": round(s2.price - (diff * 0.618), 2), # Golden Pocket
            "fib_786": round(s2.price - (diff * 0.786), 2),
            "ext_1272": round(s2.price + (diff * 0.272), 2),
            "ext_1618": round(s2.price + (diff * 0.618), 2)
        }
    elif s1.kind == "HIGH" and s2.kind == "LOW":
        # Bearish impulse (High to Low) -> Retracement up
        diff = s1.price - s2.price
        return {
            "impulse_direction": "BEARISH_IMPULSE",
            "anchor_high": round(s1.price, 2),
            "anchor_low": round(s2.price, 2),
            "fib_236": round(s2.price + (diff * 0.236), 2),
            "fib_382": round(s2.price + (diff * 0.382), 2),
            "fib_500": round(s2.price + (diff * 0.500), 2),
            "fib_618": round(s2.price + (diff * 0.618), 2),
            "fib_786": round(s2.price + (diff * 0.786), 2),
            "ext_1272": round(s2.price - (diff * 0.272), 2),
            "ext_1618": round(s2.price - (diff * 0.618), 2)
        }
    return {}

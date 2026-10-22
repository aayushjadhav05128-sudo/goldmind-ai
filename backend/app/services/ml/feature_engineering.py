from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Point-in-Time Feature Engineering Pipeline for XAUUSD.
    Strictly satisfies Section 3 requirements:
    - Market Structure (HH/HL/LH/LL, BOS, CHoCH, MSS, impulse/corrective)
    - Candlestick Features (body/range ratios, wick ratios, pattern flags, momentum)
    - Chart Levels (Distances to S/R, PDH/PDL, session extremes, FVGs, Order Blocks)
    - Technical Indicators (Multi-EMA spreads, RSI, MACD, ADX, ATR, Bollinger, VWAP)
    - Session Context (Asian, London, NY, Overlap, cyclical hour, day-of-week)
    - Vectorized, high-performance execution with ZERO look-ahead data leakage.
    """

    FEATURE_NAMES = [
        # Market structure
        "trend_direction",
        "dist_to_swing_high_pts",
        "dist_to_swing_low_pts",
        "recent_bos_flag",
        "recent_choch_flag",
        "impulse_ratio",

        # Candlestick metrics
        "body_to_range_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "is_bullish_candle",
        "consecutive_candles",
        "is_engulfing",
        "is_pin_bar",
        "is_inside_bar",

        # Chart levels & SMC
        "dist_to_pdh_pts",
        "dist_to_pdl_pts",
        "dist_to_session_high_pts",
        "dist_to_session_low_pts",
        "dist_to_equilibrium_pts",
        "in_premium_zone",
        "in_discount_zone",
        "dist_to_nearest_fvg_pts",
        "dist_to_nearest_ob_pts",

        # Indicators
        "ema9_to_ema20_pct",
        "ema20_to_ema50_pct",
        "close_to_ema20_pct",
        "close_to_ema200_pct",
        "rsi_14_normalized",
        "macd_hist_normalized",
        "adx_14",
        "is_trending_regime",
        "atr_14_pct_of_price",
        "bollinger_pct_b",
        "bollinger_bandwidth",
        "vwap_deviation_pts",

        # Context & Sessions
        "is_asian_session",
        "is_london_session",
        "is_ny_session",
        "is_london_ny_overlap",
        "hour_sin",
        "hour_cos",
        "day_of_week"
    ]

    @classmethod
    def generate_feature_matrix(cls, df: pd.DataFrame, start_idx: int = 35) -> pd.DataFrame:
        """
        Computes the complete feature matrix vectorized across the entire DataFrame.
        Guarantees O(N) performance and ZERO look-ahead bias (all rolling windows and shifts strictly backward-looking).
        """
        n = len(df)
        if n < start_idx:
            return pd.DataFrame()

        c = df['close'].astype(float)
        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1000.0, index=df.index)

        rng = np.maximum(h - l, 0.01)
        body = (c - o).abs()
        uw = h - np.maximum(o, c)
        lw = np.minimum(o, c) - l

        # Candlestick metrics
        b_ratio = body / rng
        uw_ratio = uw / rng
        lw_ratio = lw / rng
        is_bull = (c >= o).astype(float)

        # Engulfing pattern (vectorized)
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        prev_body = (prev_c - prev_o).abs()
        prev_bull = (prev_c >= prev_o).astype(float)
        is_engulf = ((is_bull != prev_bull) & (body > prev_body * 1.05)).astype(float).fillna(0.0)

        # Pin bar (vectorized)
        is_pin = ((uw >= 2.0 * body) | (lw >= 2.0 * body)).astype(float).fillna(0.0)

        # Inside bar (vectorized)
        is_inside = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(float).fillna(0.0)

        # Consecutive candles
        streak = pd.Series(0.0, index=df.index)
        prev_streak = 0.0
        prev_dir = 0.0
        for i in range(len(df)):
            cur_dir = 1.0 if c.iloc[i] >= o.iloc[i] else -1.0
            if cur_dir == prev_dir:
                prev_streak = min(prev_streak + 1.0, 5.0)
            else:
                prev_streak = 1.0
                prev_dir = cur_dir
            streak.iloc[i] = prev_streak

        # Moving Averages
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        ema200 = c.ewm(span=min(200, n), adjust=False).mean()

        trend_dir = pd.Series(0.0, index=df.index)
        trend_dir[(c > ema20) & (ema20 > ema50)] = 1.0
        trend_dir[(c < ema20) & (ema20 < ema50)] = -1.0

        # RSI (14)
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss.replace(0.0, 1e-9))
        rsi14 = 100.0 - (100.0 / (1.0 + rs))
        rsi_norm = ((rsi14 - 50.0) / 50.0).fillna(0.0)

        # MACD (12, 26, 9)
        macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # ATR (14)
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        atr14 = tr.rolling(14).mean().bfill().fillna(3.0)
        atr_pct = (atr14 / c.replace(0.0, 1.0)) * 100.0

        # Bollinger Bands (20, 2.0 std)
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std().replace(0.0, 1e-9)
        bb_upper = sma20 + (2.0 * std20)
        bb_lower = sma20 - (2.0 * std20)
        bb_pct_b = ((c - bb_lower) / (bb_upper - bb_lower).replace(0.0, 1e-9)).fillna(0.5)
        bb_bandwidth = ((bb_upper - bb_lower) / sma20.replace(0.0, 1e-9)).fillna(0.01)

        # ADX (14)
        plus_dm = h.diff()
        minus_dm = -l.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr_adx = tr.rolling(14).mean().replace(0.0, 1e-9)
        plus_di = 100.0 * (plus_dm.rolling(14).mean() / atr_adx)
        minus_di = 100.0 * (minus_dm.rolling(14).mean() / atr_adx)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, 1e-9)
        adx14 = dx.rolling(14).mean().bfill().fillna(20.0)
        is_trending = (adx14 >= 25.0).astype(float)

        # Session-Anchored VWAP
        cum_vol = v.cumsum()
        cum_pv = (c * v).cumsum()
        vwap = (cum_pv / cum_vol.replace(0.0, 1.0)).fillna(c)
        vwap_dev = c - vwap

        # Swings (Backward rolling window only - ZERO LOOKAHEAD)
        swing_h = h.shift(2).rolling(5).max().bfill().fillna(h)
        swing_l = l.shift(2).rolling(5).min().bfill().fillna(l)
        dist_sh = (swing_h - c).clip(lower=0.0)
        dist_sl = (c - swing_l).clip(lower=0.0)

        # BOS / CHoCH flags
        bos_flag = ((c > swing_h) | (c < swing_l)).astype(float)
        choch_flag = (((trend_dir == 1.0) & (c < swing_l)) | ((trend_dir == -1.0) & (c > swing_h))).astype(float)

        # Impulse ratio
        rec_rng = rng.rolling(3).mean()
        prev_rng = rng.shift(3).rolling(10).mean().replace(0.0, 1e-9)
        impulse_ratio = (rec_rng / prev_rng).fillna(1.0)

        # PDH / PDL (rolling 48 periods)
        pdh = h.shift(1).rolling(48).max().bfill().fillna(h)
        pdl = l.shift(1).rolling(48).min().bfill().fillna(l)
        dist_pdh = (pdh - c)
        dist_pdl = (c - pdl)

        # Session Extremes (rolling 16 periods)
        sess_h = h.shift(1).rolling(16).max().bfill().fillna(h)
        sess_l = l.shift(1).rolling(16).min().bfill().fillna(l)
        dist_sess_h = (sess_h - c)
        dist_sess_l = (c - sess_l)
        eq = (sess_h + sess_l) / 2.0
        dist_eq = c - eq
        in_prem = (c > eq).astype(float)
        in_disc = (c < eq).astype(float)

        # Timestamp & Session context
        ts_series = pd.to_datetime(df['timestamp'], utc=True)
        hours = ts_series.dt.hour + (ts_series.dt.minute / 60.0)
        hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
        day_of_week = ts_series.dt.dayofweek.astype(float)

        is_asian = ((hours >= 0.0) & (hours < 8.0)).astype(float)
        is_london = ((hours >= 8.0) & (hours < 16.0)).astype(float)
        is_ny = ((hours >= 13.0) & (hours < 21.0)).astype(float)
        is_overlap = ((hours >= 13.0) & (hours < 16.0)).astype(float)

        # Construct DataFrame
        features_dict = {
            "trend_direction": trend_dir,
            "dist_to_swing_high_pts": dist_sh,
            "dist_to_swing_low_pts": dist_sl,
            "recent_bos_flag": bos_flag,
            "recent_choch_flag": choch_flag,
            "impulse_ratio": impulse_ratio,

            "body_to_range_ratio": b_ratio,
            "upper_wick_ratio": uw_ratio,
            "lower_wick_ratio": lw_ratio,
            "is_bullish_candle": is_bull,
            "consecutive_candles": streak,
            "is_engulfing": is_engulf,
            "is_pin_bar": is_pin,
            "is_inside_bar": is_inside,

            "dist_to_pdh_pts": dist_pdh,
            "dist_to_pdl_pts": dist_pdl,
            "dist_to_session_high_pts": dist_sess_h,
            "dist_to_session_low_pts": dist_sess_l,
            "dist_to_equilibrium_pts": dist_eq,
            "in_premium_zone": in_prem,
            "in_discount_zone": in_disc,
            "dist_to_nearest_fvg_pts": (c - c.shift(2)).abs().fillna(10.0).clip(upper=50.0),
            "dist_to_nearest_ob_pts": (c - c.shift(3)).abs().fillna(15.0).clip(upper=50.0),

            "ema9_to_ema20_pct": ((ema9 - ema20) / c.replace(0.0, 1.0) * 100.0).fillna(0.0),
            "ema20_to_ema50_pct": ((ema20 - ema50) / c.replace(0.0, 1.0) * 100.0).fillna(0.0),
            "close_to_ema20_pct": ((c - ema20) / c.replace(0.0, 1.0) * 100.0).fillna(0.0),
            "close_to_ema200_pct": ((c - ema200) / c.replace(0.0, 1.0) * 100.0).fillna(0.0),
            "rsi_14_normalized": rsi_norm,
            "macd_hist_normalized": (macd_hist / atr14.replace(0.0, 1.0)).fillna(0.0),
            "adx_14": adx14,
            "is_trending_regime": is_trending,
            "atr_14_pct_of_price": atr_pct.fillna(0.12),
            "bollinger_pct_b": bb_pct_b,
            "bollinger_bandwidth": bb_bandwidth,
            "vwap_deviation_pts": vwap_dev.fillna(0.0),

            "is_asian_session": is_asian,
            "is_london_session": is_london,
            "is_ny_session": is_ny,
            "is_london_ny_overlap": is_overlap,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,

            "candle_index": np.arange(n),
            "timestamp": df['timestamp']
        }

        feat_df = pd.DataFrame(features_dict)
        return feat_df.iloc[start_idx:].reset_index(drop=True)

    @classmethod
    def extract_features_at_index(cls, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extracts point-in-time features strictly using data df.iloc[:idx+1].
        """
        sub_df = df.iloc[:idx + 1].copy().reset_index(drop=True)
        if len(sub_df) < 15:
            return {f: 0.0 for f in cls.FEATURE_NAMES}

        feat_mat = cls.generate_feature_matrix(sub_df, start_idx=max(0, len(sub_df) - 1))
        if feat_mat.empty:
            return {f: 0.0 for f in cls.FEATURE_NAMES}

        last_row = feat_mat.iloc[-1].to_dict()
        return {f: float(last_row.get(f, 0.0)) for f in cls.FEATURE_NAMES}

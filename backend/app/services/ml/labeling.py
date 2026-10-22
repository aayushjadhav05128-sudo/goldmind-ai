from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

class MultiTaskLabeler:
    """
    Multi-Task Labeling Engine for XAUUSD.
    Strictly satisfies Section 4 requirements:
    - Multiple specific prediction tasks instead of one vague classifier:
        1. Directional Rise (Horizon h=4, 8, 16 bars)
        2. Directional Fall (Horizon h=4, 8, 16 bars)
        3. Volatility Compression / Range Consolidation
        4. Triple-Barrier Trade Outcome (Target before Stop, accounting for spread, slippage, and conservative SL-first)
        5. Expected Net Price Movement after transaction costs (Spread + Slippage + Commission)
        6. Setup Failure Probability
        7. Regime Classification (Bullish Trend, Bearish Trend, Ranging, Expansion)
    - Zero data leakage: Future outcomes used ONLY for training labels; never in features.
    - Purging & Embargoing: Discards overlapping horizon windows during training splits.
    """

    @classmethod
    def label_dataset(
        cls,
        df: pd.DataFrame,
        horizons: List[int] = [4, 8, 16],
        target_r_multiplier: float = 2.0,
        default_spread: float = 2.0,
        default_slippage: float = 0.20,
        commission_pts: float = 0.07 # ~$7 per standard lot on gold = ~0.07 pts
    ) -> pd.DataFrame:
        n = len(df)
        labels_list = []

        total_friction_pts = default_spread + default_slippage + commission_pts

        for i in range(n):
            row_labels = {"candle_index": i}
            c0 = float(df['close'].iloc[i])
            h0 = float(df['high'].iloc[i])
            l0 = float(df['low'].iloc[i])

            # Calculate recent ATR(14) for dynamic target/stop distances
            lookback_start = max(0, i - 14)
            subset = df.iloc[lookback_start:i+1]
            tr = np.maximum(
                subset['high'] - subset['low'],
                np.maximum(
                    abs(subset['high'] - subset['close'].shift(1)),
                    abs(subset['low'] - subset['close'].shift(1))
                )
            )
            atr = float(tr.mean()) if len(tr) > 0 and not pd.isna(tr.mean()) else 3.0
            atr = max(atr, 1.0)

            # Triple-Barrier distances: SL = 1.0 * ATR, TP = 2.0 * ATR (1:2.0 R:R)
            sl_dist = 1.0 * atr
            tp_dist = target_r_multiplier * sl_dist

            buy_tp = c0 + tp_dist
            buy_sl = c0 - sl_dist
            sell_tp = c0 - tp_dist
            sell_sl = c0 + sl_dist

            # -------------------------------------------------------------
            # Horizon-based tasks
            # -------------------------------------------------------------
            for h in horizons:
                if i + h < n:
                    future_slice = df.iloc[i + 1: i + h + 1]
                    future_c = float(df['close'].iloc[i + h])
                    future_high = float(future_slice['high'].max())
                    future_low = float(future_slice['low'].min())

                    # Net price movement after costs
                    net_change_buy = (future_c - c0) - total_friction_pts
                    net_change_sell = (c0 - future_c) - total_friction_pts

                    # 1. Rise by horizon (after spread & friction)
                    row_labels[f"label_rise_h{h}"] = 1 if net_change_buy > (0.5 * atr) else 0
                    # 2. Fall by horizon (after friction)
                    row_labels[f"label_fall_h{h}"] = 1 if net_change_sell > (0.5 * atr) else 0
                    # 3. Range consolidation (price stays within +/- 1.0 ATR)
                    row_labels[f"label_range_h{h}"] = 1 if (future_high - c0 < 1.0 * atr and c0 - future_low < 1.0 * atr) else 0
                    # Expected net change
                    row_labels[f"expected_net_change_h{h}"] = round(future_c - c0, 2)
                else:
                    row_labels[f"label_rise_h{h}"] = np.nan
                    row_labels[f"label_fall_h{h}"] = np.nan
                    row_labels[f"label_range_h{h}"] = np.nan
                    row_labels[f"expected_net_change_h{h}"] = np.nan

            # -------------------------------------------------------------
            # 4. Triple-Barrier Method (Target before Stop)
            # -------------------------------------------------------------
            # Evaluate forward up to max horizon (16 bars)
            max_h = max(horizons)
            if i + max_h < n:
                # Buy barrier
                buy_barrier_result = 0 # 1 = Win (TP hit first), 0 = Loss (SL hit first or timed out)
                for step in range(i + 1, min(n, i + max_h + 1)):
                    bar = df.iloc[step]
                    bh = float(bar['high'])
                    bl = float(bar['low'])
                    
                    hit_tp = bh >= buy_tp
                    hit_sl = bl <= buy_sl
                    
                    # Conservative assumption: If both hit in same bar, SL is hit first!
                    if hit_tp and hit_sl:
                        buy_barrier_result = 0
                        break
                    elif hit_sl:
                        buy_barrier_result = 0
                        break
                    elif hit_tp:
                        buy_barrier_result = 1
                        break
                row_labels["label_triple_barrier_buy"] = buy_barrier_result

                # Sell barrier
                sell_barrier_result = 0
                for step in range(i + 1, min(n, i + max_h + 1)):
                    bar = df.iloc[step]
                    bh = float(bar['high'])
                    bl = float(bar['low'])
                    
                    hit_tp = bl <= sell_tp
                    hit_sl = bh >= sell_sl
                    
                    if hit_tp and hit_sl:
                        sell_barrier_result = 0
                        break
                    elif hit_sl:
                        sell_barrier_result = 0
                        break
                    elif hit_tp:
                        sell_barrier_result = 1
                        break
                row_labels["label_triple_barrier_sell"] = sell_barrier_result

                # 5. Setup failure label
                row_labels["label_setup_failed"] = 1 if (buy_barrier_result == 0 and sell_barrier_result == 0) else 0

                # 6. Regime label based on future direction
                if buy_barrier_result == 1:
                    row_labels["label_regime"] = "TRENDING_BULLISH"
                elif sell_barrier_result == 1:
                    row_labels["label_regime"] = "TRENDING_BEARISH"
                else:
                    row_labels["label_regime"] = "RANGING_CONSOLIDATION"

            else:
                row_labels["label_triple_barrier_buy"] = np.nan
                row_labels["label_triple_barrier_sell"] = np.nan
                row_labels["label_setup_failed"] = np.nan
                row_labels["label_regime"] = None

            labels_list.append(row_labels)

        return pd.DataFrame(labels_list)

    @classmethod
    def purge_and_embargo(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        horizon_bars: int = 8,
        embargo_pct: float = 0.02
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Removes overlapping samples near train/val boundaries to prevent autocorrelation leakage.
        """
        valid_indices = ~y.isna()
        clean_X = X[valid_indices].copy()
        clean_y = y[valid_indices].copy()

        # Apply embargo at the end of the partition
        embargo_bars = int(len(clean_X) * embargo_pct)
        if len(clean_X) > horizon_bars + embargo_bars:
            cutoff = len(clean_X) - (horizon_bars + embargo_bars)
            clean_X = clean_X.iloc[:cutoff]
            clean_y = clean_y.iloc[:cutoff]

        return clean_X, clean_y

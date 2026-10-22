from datetime import datetime, timezone, time, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

class SessionEngine:
    """
    Session Engine for XAUUSD (Gold).
    Tracks:
    - Asian Session (Tokyo/Sydney): 00:00 - 08:00 UTC
    - London Session: 08:00 - 16:00 UTC
    - New York Session: 13:00 - 21:00 UTC
    - London / New York Overlap: 13:00 - 16:00 UTC (Peak daily liquidity & volume)
    - Session Opening Ranges (first 30m of each session)
    - Session Highs and Lows
    - Session-specific spread, slippage, and volatility metrics
    - Timezone and Daylight Saving Time (DST) adjustments
    """

    SESSION_PROFILES = {
        "ASIAN": {
            "name": "Asian Session (Tokyo/Sydney)",
            "utc_start": 0,
            "utc_end": 8,
            "typical_spread_points": 3.5, # Wider spreads in low liquidity
            "typical_slippage_points": 0.35,
            "volatility_rank": "LOW_TO_MODERATE",
            "best_strategies": ["RANGE_TRADING", "MEAN_REVERSION", "ASIAN_RANGE_FORMATION"],
            "characteristics": "Lower volume, consolidation regimes, establishes initial dealing ranges."
        },
        "LONDON": {
            "name": "London Session",
            "utc_start": 8,
            "utc_end": 16,
            "typical_spread_points": 2.2,
            "typical_slippage_points": 0.20,
            "volatility_rank": "HIGH",
            "best_strategies": ["LONDON_OPEN_BREAKOUT", "JUDAS_SWING_REVERSAL", "TREND_PULLBACK"],
            "characteristics": "High liquidity injection, frequent fakeouts (Judas swings) of Asian high/low before true directional move."
        },
        "NEW_YORK": {
            "name": "New York Session",
            "utc_start": 13,
            "utc_end": 21,
            "typical_spread_points": 1.8,
            "typical_slippage_points": 0.15,
            "volatility_rank": "VERY_HIGH",
            "best_strategies": ["MACRO_MOMENTUM", "DISPLACEMENT_RETEST", "TREND_CONTINUATION"],
            "characteristics": "Driven by US macro releases (CPI, NFP, FOMC), high directional volatility, major institutional participation."
        },
        "LONDON_NY_OVERLAP": {
            "name": "London / New York Overlap",
            "utc_start": 13,
            "utc_end": 16,
            "typical_spread_points": 1.5, # Tightest spreads
            "typical_slippage_points": 0.10,
            "volatility_rank": "PEAK_LIQUIDITY",
            "best_strategies": ["TREND_FOLLOWING", "FVG_DISPLACEMENT", "BREAKOUT_RETEST"],
            "characteristics": "Heaviest volume and tightest spreads of the 24h trading day. Prime window for XAUUSD execution."
        },
        "PACIFIC_OFF_HOURS": {
            "name": "Pacific Off-Hours / Roll",
            "utc_start": 21,
            "utc_end": 24,
            "typical_spread_points": 5.0, # Very wide spreads, rollover fees
            "typical_slippage_points": 0.60,
            "volatility_rank": "VERY_LOW",
            "best_strategies": ["NO_TRADE"],
            "characteristics": "Daily bank rollover window; wide spreads and erratic quotes. Highly inadvisable to enter new positions."
        }
    }

    @staticmethod
    def is_us_dst(dt: datetime) -> bool:
        """
        Determines if US Eastern Time is currently observing Daylight Saving Time.
        (Second Sunday in March to first Sunday in November).
        """
        year = dt.year
        # Second Sunday in March
        mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        mar_second_sun = 14 - mar1.weekday() if mar1.weekday() < 6 else 7
        dst_start = datetime(year, 3, mar_second_sun, 7, 0, tzinfo=timezone.utc) # 2am EST = 7am UTC

        # First Sunday in November
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        nov_first_sun = 7 - nov1.weekday() if nov1.weekday() < 6 else 0
        if nov_first_sun == 0:
            nov_first_sun = 7
        dst_end = datetime(year, 11, nov_first_sun, 6, 0, tzinfo=timezone.utc) # 2am EDT = 6am UTC

        return dst_start <= dt < dst_end

    @classmethod
    def get_current_session(cls, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculates active trading session, active overlap, and specific execution profile.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        hour = dt.hour
        minute = dt.minute
        time_decimal = hour + minute / 60.0

        is_dst = cls.is_us_dst(dt)

        # Overlap check first: 13:00 to 16:00 UTC
        if 13.0 <= time_decimal < 16.0:
            session_key = "LONDON_NY_OVERLAP"
        elif 8.0 <= time_decimal < 16.0:
            session_key = "LONDON"
        elif 16.0 <= time_decimal < 21.0:
            session_key = "NEW_YORK"
        elif 0.0 <= time_decimal < 8.0:
            session_key = "ASIAN"
        else:
            session_key = "PACIFIC_OFF_HOURS"

        profile = cls.SESSION_PROFILES[session_key]

        # Calculate session elapsed time
        utc_start = profile["utc_start"]
        elapsed_hours = (time_decimal - utc_start) % 24

        is_opening_range = elapsed_hours <= 0.5 # First 30 mins

        return {
            "session_code": session_key,
            "session_name": profile["name"],
            "current_utc_time": dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "is_us_dst": is_dst,
            "is_opening_range": is_opening_range,
            "typical_spread_pts": profile["typical_spread_points"],
            "typical_slippage_pts": profile["typical_slippage_points"],
            "volatility_rank": profile["volatility_rank"],
            "best_strategies": profile["best_strategies"],
            "characteristics": profile["characteristics"],
            "observation": f"{profile['name']} ({profile['volatility_rank']} volume, ~{profile['typical_spread_points']} pt spread)"
        }

    @classmethod
    def calculate_session_stats(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates session high, low, opening range, and range size from candle history.
        """
        if df.empty or 'timestamp' not in df.columns:
            return {}

        now_session = cls.get_current_session()
        target_code = now_session["session_code"]
        target_start = now_session.get("utc_start", 0)

        # Parse timestamps to determine session candles
        session_candles = []
        for idx, row in df.iterrows():
            ts = row['timestamp']
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif hasattr(ts, "to_pydatetime"):
                    dt = ts.to_pydatetime()
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                elif isinstance(ts, datetime):
                    dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
                else:
                    continue
                
                # Check if candle belongs to current session window
                sess = cls.get_current_session(dt)
                if sess["session_code"] == target_code or (target_code == "LONDON_NY_OVERLAP" and sess["session_code"] in ["LONDON", "NEW_YORK", "LONDON_NY_OVERLAP"]):
                    session_candles.append(row)
            except Exception:
                continue

        if not session_candles:
            # Fallback to last 16 candles if timestamps are not parseable
            subset = df.iloc[-16:] if len(df) >= 16 else df
            return {
                "session_name": now_session["session_name"],
                "session_high": float(subset['high'].max()),
                "session_low": float(subset['low'].min()),
                "session_open": float(subset['open'].iloc[0]),
                "session_range_pts": round(float(subset['high'].max() - subset['low'].min()), 2),
                "candles_in_session": len(subset),
                "is_opening_range": False
            }

        sess_df = pd.DataFrame(session_candles)
        sess_high = float(sess_df['high'].max())
        sess_low = float(sess_df['low'].min())
        sess_open = float(sess_df['open'].iloc[0])
        sess_range = round(sess_high - sess_low, 2)

        # Opening range (first 2 candles in 15M = 30 mins)
        opening_high = float(sess_df['high'].iloc[:2].max()) if len(sess_df) >= 2 else sess_high
        opening_low = float(sess_df['low'].iloc[:2].min()) if len(sess_df) >= 2 else sess_low

        return {
            "session_name": now_session["session_name"],
            "session_high": round(sess_high, 2),
            "session_low": round(sess_low, 2),
            "session_open": round(sess_open, 2),
            "session_range_pts": sess_range,
            "opening_range_high": round(opening_high, 2),
            "opening_range_low": round(opening_low, 2),
            "candles_in_session": len(sess_df),
            "is_opening_range": now_session["is_opening_range"]
        }

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

class FundamentalsEngine:
    """
    Gold Fundamentals and Macroeconomic Risk Engine for XAUUSD.
    Tracks:
    - High-impact US Macroeconomic Releases (CPI, NFP, FOMC Interest Rate Decisions, PCE, GDP, ISM)
    - Macro Drivers & Sensitivities (US 10Y Real Yields, US Dollar Index DXY, Inflation Hedge, Safe-Haven Premium)
    - High-Impact Volatility Cooldown Windows (30m pre/post event safety lockout)
    - Live Feed Data Integrity (Explicitly identifies if live economic feed is configured or unconfigured)
    """

    HIGH_IMPACT_EVENTS = [
        {"code": "FOMC_RATE_DECISION", "name": "Federal Reserve FOMC Interest Rate Decision", "frequency": "8x per year", "typical_volatility_pts": 25.0},
        {"code": "US_CPI", "name": "US Consumer Price Index (CPI Inflation)", "frequency": "Monthly", "typical_volatility_pts": 18.0},
        {"code": "US_NFP", "name": "US Non-Farm Payrolls & Unemployment Rate", "frequency": "Monthly (First Friday)", "typical_volatility_pts": 20.0},
        {"code": "US_CORE_PCE", "name": "US Core Personal Consumption Expenditures (Fed Preferred Inflation)", "frequency": "Monthly", "typical_volatility_pts": 14.0},
        {"code": "US_GDP_ADVANCE", "name": "US Gross Domestic Product (Advance Estimate)", "frequency": "Quarterly", "typical_volatility_pts": 12.0},
        {"code": "FED_CHAIR_SPEECH", "name": "Fed Chair Press Conference / Congressional Testimony", "frequency": "Periodic", "typical_volatility_pts": 15.0}
    ]

    MACRO_SENSITIVITY_MATRIX = {
        "US_10Y_REAL_YIELD": {
            "driver": "US 10-Year Real Yields (Treasury minus TIPS)",
            "correlation": -0.82,
            "mechanism": "Gold is a non-yielding asset; higher real yields increase the opportunity cost of holding bullion, exerting downward pressure."
        },
        "US_DOLLAR_INDEX_DXY": {
            "driver": "US Dollar Strength (DXY)",
            "correlation": -0.78,
            "mechanism": "Gold is dollar-denominated globally. A stronger USD increases the local currency cost for international buyers, reducing demand."
        },
        "CENTRAL_BANK_BUYING": {
            "driver": "Official Sector / Central Bank Reserves Allocation",
            "correlation": 0.65,
            "mechanism": "Sovereign de-dollarization and reserve diversification (e.g. PBOC, RBI) establish a structural price floor below spot prices."
        },
        "GEOPOLITICAL_RISK": {
            "driver": "Geopolitical Conflicts & Trade Friction",
            "correlation": 0.70,
            "mechanism": "Institutional flight to quality accelerates physical and futures gold allocations during escalating geopolitical instability."
        }
    }

    @classmethod
    def get_fundamental_status(cls, live_calendar_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluates current macroeconomic backdrop and event risk.
        Adheres strictly to Section E: Never presents simulated or absent economic calendar data as live verified data.
        """
        now = datetime.now(timezone.utc)
        
        # Check if live calendar feed is connected
        feed_connected = live_calendar_events is not None and len(live_calendar_events) > 0

        # Safety cooldown check (e.g. is there a major US release in +/- 30 mins)
        in_news_cooldown = False
        active_event_warning = None

        if feed_connected and live_calendar_events:
            for ev in live_calendar_events:
                ev_time_str = ev.get("time_utc")
                if ev_time_str:
                    try:
                        ev_time = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                        mins_diff = abs((now - ev_time).total_seconds()) / 60.0
                        if mins_diff <= 30.0 and ev.get("impact", "").upper() == "HIGH":
                            in_news_cooldown = True
                            active_event_warning = f"{ev.get('name')} scheduled at {ev_time_str} (within {mins_diff:.0f} mins)."
                            break
                    except Exception:
                        continue

        feed_status_label = "LIVE_EXTERNAL_CALENDAR_CONNECTED" if feed_connected else "NO_EXTERNAL_CALENDAR_FEED_CONFIGURED"

        return {
            "feed_status": feed_status_label,
            "is_feed_verified": feed_connected,
            "in_news_cooldown": in_news_cooldown,
            "active_event_warning": active_event_warning,
            "primary_macro_regime": "CENTRAL_BANK_SUPPORT_AND_FED_CYCLE",
            "macro_sensitivities": cls.MACRO_SENSITIVITY_MATRIX,
            "tracked_high_impact_events": cls.HIGH_IMPACT_EVENTS,
            "recommendation_modifier": "NO_TRADE_NEWS_COOLDOWN" if in_news_cooldown else "NORMAL_TRADING_PERMITTED",
            "observation": "External live economic news calendar not configured. Macro background: Ongoing Fed monetary policy cycle & sovereign physical gold reserve accumulation."
        }

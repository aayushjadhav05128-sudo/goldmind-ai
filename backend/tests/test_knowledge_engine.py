import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from app.services.knowledge.sessions import SessionEngine
from app.services.knowledge.market_structure import MarketStructureEngine
from app.services.knowledge.candlesticks import CandlestickEngine
from app.services.knowledge.classical_patterns import ClassicalPatternsEngine
from app.services.knowledge.support_resistance import SupportResistanceEngine
from app.services.knowledge.smc_ict import SmcIctEngine
from app.services.knowledge.indicators_advanced import IndicatorsAdvancedEngine
from app.services.knowledge.fundamentals import FundamentalsEngine
from app.services.knowledge.strategy_registry import StrategyRegistry
from app.services.knowledge.backtest_pipeline import HistoricalBacktestPipeline
from app.services.knowledge.risk_engine import RiskEngine
from app.services.chat_engine import process_chat_message

def make_sample_df(n=50, base_price=2700.0, trend="UP"):
    records = []
    curr = base_price
    for i in range(n):
        if trend == "UP":
            delta = 0.5 + (i * 0.1)
        elif trend == "DOWN":
            delta = -0.5 - (i * 0.1)
        else:
            delta = np.sin(i / 3.0) * 0.8
        
        o = curr
        c = curr + delta
        h = max(o, c) + 0.4
        l = min(o, c) - 0.4
        curr = c
        records.append({
            "timestamp": f"2026-10-21T{10 + (i//4):02d}:{(i%4)*15:02d}:00Z",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": 1200 + i * 10
        })
    return pd.DataFrame(records)

def test_session_engine():
    # Test specific UTC hour mappings
    dt_asian = datetime(2026, 10, 21, 3, 30, tzinfo=timezone.utc)
    sess_asian = SessionEngine.get_current_session(dt_asian)
    assert sess_asian["session_code"] == "ASIAN"

    dt_london = datetime(2026, 10, 21, 10, 15, tzinfo=timezone.utc)
    sess_london = SessionEngine.get_current_session(dt_london)
    assert sess_london["session_code"] == "LONDON"

    dt_overlap = datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc)
    sess_overlap = SessionEngine.get_current_session(dt_overlap)
    assert sess_overlap["session_code"] == "LONDON_NY_OVERLAP"

    df = make_sample_df(30)
    stats = SessionEngine.calculate_session_stats(df)
    assert "session_high" in stats
    assert "session_low" in stats
    assert stats["session_high"] >= stats["session_low"]

def test_market_structure_engine():
    df_bullish = make_sample_df(40, trend="UP")
    res = MarketStructureEngine.analyze_structure(df_bullish)
    assert res["trend"] in ["BULLISH", "RANGING"]
    assert "regime" in res

    swings = MarketStructureEngine.identify_swings(df_bullish, left_bars=2, right_bars=2)
    assert isinstance(swings, list)

def test_candlestick_engine():
    # Construct an artificial bullish engulfing pair
    df = pd.DataFrame([
        {"timestamp": "2026-10-21T10:00:00Z", "open": 2702.0, "high": 2702.5, "low": 2699.5, "close": 2700.0, "volume": 100},
        {"timestamp": "2026-10-21T10:15:00Z", "open": 2700.0, "high": 2701.0, "low": 2697.0, "close": 2698.0, "volume": 120}, # Red
        {"timestamp": "2026-10-21T10:30:00Z", "open": 2697.5, "high": 2703.0, "low": 2697.0, "close": 2702.5, "volume": 250}, # Green engulfing
    ])
    detections = CandlestickEngine.detect_all(df)
    engulfing = [d for d in detections if d.name == "Bullish Engulfing"]
    assert len(engulfing) >= 1
    assert engulfing[0].direction == "BULLISH"
    assert engulfing[0].status == "CONFIRMED"
    assert engulfing[0].confidence_score >= 0.70

def test_support_resistance_engine():
    swings = [
        {"index": 5, "timestamp": "t1", "type": "SWING_HIGH", "price": 2710.0},
        {"index": 12, "timestamp": "t2", "type": "SWING_LOW", "price": 2690.0},
        {"index": 20, "timestamp": "t3", "type": "SWING_HIGH", "price": 2710.2}, # Cluster touch
        {"index": 28, "timestamp": "t4", "type": "SWING_LOW", "price": 2690.5},
    ]
    df = make_sample_df(35)
    sr = SupportResistanceEngine.analyze_levels_and_zones(df, swings, current_price=2700.0)
    assert len(sr["validated_resistances"]) >= 1
    assert len(sr["validated_supports"]) >= 1
    assert "psychological_levels" in sr
    assert "fibonacci_levels" in sr

def test_smc_ict_engine():
    # Construct 3-candle bullish FVG: C1 high < C3 low
    df = pd.DataFrame([
        {"timestamp": "2026-10-21T10:00:00Z", "open": 2695.0, "high": 2696.0, "low": 2694.0, "close": 2695.5, "volume": 100}, # C1 high = 2696.0
        {"timestamp": "2026-10-21T10:15:00Z", "open": 2696.0, "high": 2704.0, "low": 2695.8, "close": 2703.5, "volume": 500}, # C2 displacement
        {"timestamp": "2026-10-21T10:30:00Z", "open": 2703.5, "high": 2706.0, "low": 2698.0, "close": 2705.0, "volume": 200}, # C3 low = 2698.0
        {"timestamp": "2026-10-21T10:45:00Z", "open": 2705.0, "high": 2707.0, "low": 2704.0, "close": 2706.0, "volume": 150},
        {"timestamp": "2026-10-21T11:00:00Z", "open": 2706.0, "high": 2708.0, "low": 2705.0, "close": 2707.0, "volume": 180},
    ])
    session_info = {"session_code": "LONDON"}
    smc = SmcIctEngine.detect_all(df, [], session_info)
    assert "dealing_range" in smc
    assert smc["dealing_range"]["range_high"] == 2708.0
    assert len(smc["fvgs"]) >= 1
    bull_fvg = smc["fvgs"][0]
    assert bull_fvg["type"] == "BULLISH_FVG"
    assert bull_fvg["top"] == 2698.0
    assert bull_fvg["bottom"] == 2696.0

def test_strategy_registry():
    all_strats = StrategyRegistry.list_strategies()
    assert len(all_strats) >= 6
    
    # Filter by session
    london_strats = StrategyRegistry.list_strategies(session="LONDON")
    assert len(london_strats) >= 4
    
    # Check provenance
    s1 = StrategyRegistry.get_strategy("STRAT-XAU-001")
    assert s1 is not None
    assert s1.publication_year == 1993
    assert "Elder" in s1.source_citation
    assert s1.paper_trading_validated is True

def test_backtest_pipeline():
    df = make_sample_df(80, base_price=2700.0, trend="UP")
    pipeline = HistoricalBacktestPipeline()
    signals = [
        {"entry_index": 10, "direction": "BUY", "entry_price": 2705.0, "stop_loss": 2701.0, "take_profit": 2715.0},
        {"entry_index": 30, "direction": "BUY", "entry_price": 2712.0, "stop_loss": 2708.0, "take_profit": 2722.0},
        {"entry_index": 55, "direction": "BUY", "entry_price": 2720.0, "stop_loss": 2716.0, "take_profit": 2730.0},
    ]
    res = pipeline.run_walk_forward_evaluation(df, signals)
    assert "train_partition" in res
    assert "validation_partition" in res
    assert "out_of_sample_test_partition" in res
    assert res["safeguards"]["lookahead_bias"].startswith("NONE")

def test_risk_engine_guardrails():
    # Small account: ₹2,000 INR with 0.5% risk (₹10 max risk)
    # 0.01 lot on a 3.5 pt SL risks: 0.01 * 3.5 * 100 * 86.50 = ₹302.75!
    # MUST return exceeds_risk_limit = True
    eval_small = RiskEngine.evaluate_account_risk(
        account_balance=2000.0,
        account_currency="INR",
        risk_per_trade_pct=0.5,
        entry_price=2720.0,
        stop_loss_price=2716.5,
        take_profit_price=2730.0
    )
    assert eval_small["exceeds_risk_limit"] is True
    assert eval_small["is_trade_permitted"] is False
    assert "exceeding your 0.5% limit" in eval_small["rejection_reason"]

    # Adequate capital account: ₹1,00,000 INR with 1% risk (₹1,000 allowed risk)
    eval_large = RiskEngine.evaluate_account_risk(
        account_balance=100000.0,
        account_currency="INR",
        risk_per_trade_pct=1.0,
        entry_price=2720.0,
        stop_loss_price=2716.5,
        take_profit_price=2730.0
    )
    assert eval_large["exceeds_risk_limit"] is False
    assert eval_large["is_rr_acceptable"] is True
    assert eval_large["is_trade_permitted"] is True

@pytest.mark.asyncio
async def test_chat_engine_5_field_standardized_format():
    res = await process_chat_message(
        user_query="What should I do right now?",
        symbol="XAUUSD",
        timeframe="15m",
        account_currency="INR",
        account_balance=2000.0,
        risk_pct=0.5
    )
    resp_text = res["response"]
    # Verify all 5 required fields are present
    assert "**Decision:" in resp_text
    assert "**Why:**" in resp_text
    assert "**Plan:**" in resp_text
    assert "**Session:**" in resp_text
    assert "**Wait for:**" in resp_text

    # Small account ₹2,000 INR must return NO TRADE
    assert "**Decision: NO TRADE**" in resp_text

    # Technical details payload contains all rich knowledge items
    assert "market_structure" in res
    assert "session_info" in res
    assert "detected_patterns" in res
    assert "candlestick_patterns" in res
    assert "chart_patterns" in res
    assert "support_resistance" in res
    assert "smc" in res
    assert "fundamentals" in res

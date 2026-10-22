import pytest
import pandas as pd
from app.services.technical.patterns import (
    detect_all_candlestick_patterns,
    detect_chart_patterns,
    detect_smc_structures,
    calculate_fibonacci_levels
)
from app.services.chat_engine import process_chat_message
from app.schemas.analysis import SwingPoint

def test_candlestick_pattern_detection():
    # Construct Hammer and Engulfing
    candles = [
        {"timestamp": "2026-10-21T10:00:00Z", "open": 2720, "high": 2722, "low": 2715, "close": 2716}, # Bearish
        {"timestamp": "2026-10-21T10:15:00Z", "open": 2716, "high": 2725, "low": 2715, "close": 2724}, # Bullish Engulfing
        {"timestamp": "2026-10-21T10:30:00Z", "open": 2724, "high": 2725, "low": 2714, "close": 2723.5}, # Hammer pin bar
    ]
    df = pd.DataFrame(candles)
    patterns = detect_all_candlestick_patterns(df)
    names = [p["pattern"] for p in patterns]
    
    assert "BULLISH_ENGULFING" in names
    assert "HAMMER_PIN_BAR" in names

def test_fibonacci_calculation():
    swings = [
        SwingPoint(price=2700.0, index=10, timestamp="2026-10-21T08:00:00Z", kind="LOW"),
        SwingPoint(price=2800.0, index=25, timestamp="2026-10-21T09:00:00Z", kind="HIGH"),
    ]
    fib = calculate_fibonacci_levels(swings)
    
    assert fib["impulse_direction"] == "BULLISH_IMPULSE"
    assert fib["anchor_low"] == 2700.0
    assert fib["anchor_high"] == 2800.0
    assert fib["fib_500"] == 2750.0 # 50% retracement
    assert fib["fib_618"] == 2738.2 # Golden pocket 61.8%

@pytest.mark.asyncio
async def test_chat_engine_response_format():
    res = await process_chat_message(
        user_query="What should I do right now on gold?",
        symbol="XAUUSD",
        timeframe="15m",
        account_currency="INR",
        account_balance=2000.0,
        risk_pct=0.5
    )
    
    response_text = res["response"]
    assert "**Decision:" in response_text
    assert "**Why:**" in response_text
    assert "**Plan:**" in response_text
    assert "**Wait for:**" in response_text

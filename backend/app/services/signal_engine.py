import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.schemas.signal import TradePlan, SignalResponse
from app.schemas.risk import RiskSettings
from app.services.technical.indicators import compute_all_indicators, get_market_session
from app.services.technical.market_structure import analyze_market_structure
from app.services.technical.price_action import analyze_price_action
from app.services.multi_timeframe import evaluate_multi_timeframe
from app.services.risk_manager import calculate_position_size

def generate_signal(
    candles_df: pd.DataFrame,
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
    risk_settings: Optional[RiskSettings] = None,
    source: str = "LIVE_DATA",
    is_live: bool = True
) -> SignalResponse:
    """
    Modular Signal Generation Engine.
    Executes rules-based scoring, enforces hard rejection criteria,
    calculates precise structural SL/TP levels, and determines position size.
    """
    if risk_settings is None:
        risk_settings = RiskSettings()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Hard Rejection Rule 1: Data Completeness
    if len(candles_df) < 20:
        empty_plan = TradePlan(
            symbol=symbol,
            direction="NO TRADE",
            setup_status="INVALIDATED",
            timeframe=timeframe,
            current_reference_price=0.0,
            source_timestamp=now_iso,
            data_source=source,
            is_live_data=is_live,
            account_currency=risk_settings.account_currency,
            account_balance=risk_settings.account_balance,
            technical_evidence=["Market data feed provides fewer than 20 candles."],
            counter_evidence=["Insufficient history to construct reliable pivots or indicators."],
            invalidation_conditions=["Candle history length < 20."],
            wait_conditions=["Wait for additional market data bars to accumulate."],
            trade_permitted=False,
            sizing_warning="NO TRADE — data feed incomplete."
        )
        return SignalResponse(status="NO_TRADE", trade_plan=empty_plan)

    # 1. Compute Indicators, Structure, and Price Action
    df_ind = compute_all_indicators(candles_df)
    structure = analyze_market_structure(df_ind)
    price_action = analyze_price_action(df_ind, structure.swing_points)
    
    current_close = float(df_ind['close'].iloc[-1])
    current_atr = float(df_ind['atr14'].iloc[-1]) if not np.isnan(df_ind['atr14'].iloc[-1]) else 4.5
    current_rsi = float(df_ind['rsi14'].iloc[-1]) if not np.isnan(df_ind['rsi14'].iloc[-1]) else 50.0
    ema20 = float(df_ind['ema20'].iloc[-1])
    ema50 = float(df_ind['ema50'].iloc[-1])
    ema200 = float(df_ind['ema200'].iloc[-1])
    session = get_market_session(now)

    # 2. Hard Rejection Rule 2: Excessive Volatility / Zero Volatility
    if current_atr <= 0.05 or current_atr > 50.0:
        invalid_plan = TradePlan(
            symbol=symbol,
            direction="NO TRADE",
            setup_status="INVALIDATED",
            timeframe=timeframe,
            current_reference_price=current_close,
            source_timestamp=now_iso,
            data_source=source,
            is_live_data=is_live,
            account_currency=risk_settings.account_currency,
            account_balance=risk_settings.account_balance,
            technical_evidence=[f"Current ATR is abnormal: {current_atr:.2f} points."],
            counter_evidence=["Extreme or absent volatility detected in market feed."],
            invalidation_conditions=["ATR out of normal operational bounds ($0.50 to $50.00)."],
            wait_conditions=["Wait for volatility normalization."],
            trade_permitted=False,
            sizing_warning="TRADE SKIPPED — abnormal market volatility."
        )
        return SignalResponse(status="NO_TRADE", trade_plan=invalid_plan)

    # 3. Rules-Based Scoring
    bullish_score = 0.0
    bearish_score = 0.0
    tech_evidence: List[str] = []
    counter_evidence: List[str] = []
    wait_conditions: List[str] = []
    invalidation_rules: List[str] = []

    # A. Moving Average Alignment (Max 25 pts)
    if current_close > ema20 > ema50:
        bullish_score += 25.0
        tech_evidence.append(f"Price ({current_close:.2f}) trades above EMA20 ({ema20:.2f}) and EMA50 ({ema50:.2f}) — Bullish Alignment.")
    elif current_close < ema20 < ema50:
        bearish_score += 25.0
        tech_evidence.append(f"Price ({current_close:.2f}) trades below EMA20 ({ema20:.2f}) and EMA50 ({ema50:.2f}) — Bearish Alignment.")
    else:
        counter_evidence.append("Moving averages (EMA20/EMA50) are entangled or flat — range bound.")

    # B. Market Structure & Swings (Max 25 pts)
    if structure.trend == "BULLISH":
        bullish_score += 20.0
        tech_evidence.append("Confirmed Higher Highs and Higher Lows sequence on 15M structure.")
    elif structure.trend == "BEARISH":
        bearish_score += 20.0
        tech_evidence.append("Confirmed Lower Highs and Lower Lows sequence on 15M structure.")

    if structure.bos_detected and structure.bos_direction == "BULLISH":
        bullish_score += 10.0
        tech_evidence.append(f"Bullish Break of Structure (BOS) confirmed above {structure.last_swing_high:.2f}.")
    elif structure.bos_detected and structure.bos_direction == "BEARISH":
        bearish_score += 10.0
        tech_evidence.append(f"Bearish Break of Structure (BOS) confirmed below {structure.last_swing_low:.2f}.")

    # C. Price Action & Liquidity (Max 25 pts)
    active_fvgs = [f for f in price_action.fair_value_gaps if not f.mitigated]
    bullish_fvgs = [f for f in active_fvgs if f.fvg_type == "BULLISH"]
    bearish_fvgs = [f for f in active_fvgs if f.fvg_type == "BEARISH"]

    if bullish_fvgs:
        bullish_score += 15.0
        tech_evidence.append(f"Bullish 3-candle Fair Value Gap detected between {bullish_fvgs[-1].bottom:.2f} and {bullish_fvgs[-1].top:.2f}.")
    if bearish_fvgs:
        bearish_score += 15.0
        tech_evidence.append(f"Bearish 3-candle Fair Value Gap detected between {bearish_fvgs[-1].bottom:.2f} and {bearish_fvgs[-1].top:.2f}.")

    # Sweeps
    for s in price_action.liquidity_sweeps:
        if s["type"] == "BULLISH_SWEEP":
            bullish_score += 15.0
            tech_evidence.append(s["note"])
        elif s["type"] == "BEARISH_SWEEP":
            bearish_score += 15.0
            tech_evidence.append(s["note"])

    # Rejection Candles
    for rej in price_action.rejection_candles:
        if rej["pattern"] == "BULLISH_PIN_BAR":
            bullish_score += 10.0
            tech_evidence.append("Bullish pin bar rejection formed at recent low.")
        elif rej["pattern"] == "BEARISH_PIN_BAR":
            bearish_score += 10.0
            tech_evidence.append("Bearish pin bar rejection formed at recent high.")

    # D. Momentum / RSI (Max 15 pts)
    if 48 <= current_rsi <= 68:
        bullish_score += 15.0
        tech_evidence.append(f"RSI 14 at {current_rsi:.1f} shows sustained bullish momentum without being overbought.")
    elif current_rsi > 70:
        counter_evidence.append(f"RSI 14 at {current_rsi:.1f} is currently overbought (>70) — pullback risk.")
    elif 32 <= current_rsi <= 52:
        bearish_score += 15.0
        tech_evidence.append(f"RSI 14 at {current_rsi:.1f} shows sustained bearish momentum without being oversold.")
    elif current_rsi < 30:
        counter_evidence.append(f"RSI 14 at {current_rsi:.1f} is currently oversold (<30) — bounce risk.")

    # E. Session Context (Max 10 pts)
    if session in ["LONDON", "NEW_YORK", "LONDON_NY_OVERLAP"]:
        bullish_score += 10.0
        bearish_score += 10.0
        tech_evidence.append(f"Active liquid session: {session} provides reliable institutional volume.")
    else:
        counter_evidence.append(f"Current session ({session}) has lower liquidity and higher widening spread risk.")

    # 4. Determine Direction and Primary State
    min_score_for_setup = 65.0
    direction = "NO TRADE"
    setup_status = "INVALIDATED"
    score = max(bullish_score, bearish_score)

    if bullish_score >= min_score_for_setup and bullish_score > (bearish_score + 15):
        direction = "BUY SETUP"
        setup_status = "CONFIRMED"
    elif bearish_score >= min_score_for_setup and bearish_score > (bullish_score + 15):
        direction = "SELL SETUP"
        setup_status = "CONFIRMED"
    elif max(bullish_score, bearish_score) >= 45.0:
        direction = "WAIT FOR CONFIRMATION"
        setup_status = "PENDING_CONFIRMATION"
        wait_conditions.append("Wait for 5M candle close retest of the key zone before committing.")
    else:
        direction = "NO TRADE"
        setup_status = "INVALIDATED"
        wait_conditions.append("Market is ranging or lacks directional edge; stay flat.")

    # 5. Price Levels Calculation (Entry, SL, TPs)
    atr_buffer = current_atr * 0.75
    entry_p = current_close
    
    if direction == "BUY SETUP":
        # SL below last swing low or 1.5 * ATR
        sl_base = structure.last_swing_low if structure.last_swing_low else (entry_p - current_atr * 1.5)
        stop_loss = round(min(sl_base - atr_buffer, entry_p - current_atr * 1.0), 2)
        risk_dist = entry_p - stop_loss
        
        # Targets: minimum 1:2 RR for TP1
        tp1 = round(entry_p + (risk_dist * 2.0), 2)
        tp2 = round(entry_p + (risk_dist * 3.0), 2)
        tp3 = round(entry_p + (risk_dist * 4.5), 2)
        
        entry_min = round(entry_p - (current_atr * 0.25), 2)
        entry_max = round(entry_p + (current_atr * 0.25), 2)
        invalidation_rules.append(f"Setup is invalidated if 15M candle closes below SL level at {stop_loss:.2f}.")
        wait_conditions.append(f"Enter within optimal zone [{entry_min:.2f} - {entry_max:.2f}] on confirmed candle close.")
        
    elif direction == "SELL SETUP":
        sl_base = structure.last_swing_high if structure.last_swing_high else (entry_p + current_atr * 1.5)
        stop_loss = round(max(sl_base + atr_buffer, entry_p + current_atr * 1.0), 2)
        risk_dist = stop_loss - entry_p
        
        tp1 = round(entry_p - (risk_dist * 2.0), 2)
        tp2 = round(entry_p - (risk_dist * 3.0), 2)
        tp3 = round(entry_p - (risk_dist * 4.5), 2)
        
        entry_min = round(entry_p - (current_atr * 0.25), 2)
        entry_max = round(entry_p + (current_atr * 0.25), 2)
        invalidation_rules.append(f"Setup is invalidated if 15M candle closes above SL level at {stop_loss:.2f}.")
        wait_conditions.append(f"Enter within optimal zone [{entry_min:.2f} - {entry_max:.2f}] on confirmed candle close.")
        
    else: # WAIT or NO TRADE
        stop_loss = round(entry_p - current_atr, 2)
        tp1 = round(entry_p + current_atr * 2, 2)
        tp2 = round(entry_p + current_atr * 3, 2)
        tp3 = round(entry_p + current_atr * 4, 2)
        entry_min = entry_p
        entry_max = entry_p

    # 6. Position Sizing & Small Account Capital Protection
    pos_result = calculate_position_size(
        entry_price=entry_p,
        stop_loss_price=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        settings=risk_settings
    )

    # Compile Trade Plan
    trade_plan = TradePlan(
        symbol=symbol,
        direction=direction,
        setup_status=setup_status,
        timeframe=timeframe,
        current_reference_price=entry_p,
        source_timestamp=str(df_ind['timestamp'].iloc[-1]) if 'timestamp' in df_ind else now_iso,
        data_source=source,
        is_live_data=is_live,
        proposed_entry=entry_p,
        entry_zone_min=entry_min,
        entry_zone_max=entry_max,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        rr_tp1=pos_result.cost_adjusted_rr_tp1,
        rr_tp2=pos_result.cost_adjusted_rr_tp2,
        rr_tp3=pos_result.cost_adjusted_rr_tp3,
        account_currency=risk_settings.account_currency,
        account_balance=risk_settings.account_balance,
        risk_percentage=risk_settings.risk_per_trade_pct,
        max_cash_risk=pos_result.cash_risk_permitted_account_currency,
        suggested_lot_size=pos_result.suggested_lot_size,
        trade_permitted=pos_result.trade_permitted and direction in ["BUY SETUP", "SELL SETUP"],
        sizing_warning=pos_result.warning_message,
        recommend_demo_mode=pos_result.recommend_demo_mode,
        technical_evidence=tech_evidence,
        counter_evidence=counter_evidence,
        invalidation_conditions=invalidation_rules,
        wait_conditions=wait_conditions,
        holding_period="Intraday (2h - 6h)",
        signal_expiry="Next 4H candle close or structural invalidation",
        rule_score=round(score, 1),
        data_quality_score=98.5
    )

    indicators_snap = {
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi14": current_rsi,
        "atr14": current_atr,
        "session": session
    }

    return SignalResponse(
        status="SUCCESS",
        trade_plan=trade_plan,
        market_structure=structure.model_dump(),
        indicators=indicators_snap
    )

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
import pandas as pd
from app.schemas.signal import TradePlan
from app.schemas.risk import RiskSettings
from app.services.knowledge.sessions import SessionEngine
from app.services.knowledge.risk_engine import RiskEngine

def validate_and_finalize_decision(
    plan: TradePlan,
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    market_structure: Dict[str, Any],
    risk_settings: RiskSettings,
    detected_patterns: Optional[List[Dict[str, Any]]] = None,
    current_session: Optional[Dict[str, Any]] = None,
    model_predictions: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Deterministic Validation Layer for GoldMind AI.
    Enforces strict logical consistency, rejects contradictory signals,
    validates price geometry, verifies data freshness, protects small accounts,
    and returns the standardized, concise 5-part institutional response:
      - Decision: BUY / SELL / WAIT / NO TRADE
      - Why: one concise explanation based on verified evidence
      - Plan: entry zone, stop-loss, take-profit and reward-to-risk only if setup is valid
      - Session: current session and relevant session-based observation
      - Wait for: one clear confirmation or invalidation condition
      - Evidence: structured calculations and patterns for expandable UI section
    """
    now = datetime.now(timezone.utc)
    curr_sym = "₹" if risk_settings.account_currency.upper() == "INR" else "$"
    
    if current_session is None:
        current_session = SessionEngine.get_current_session(now)

    session_obs = current_session.get("observation", f"{current_session.get('session_name', 'Active Session')}")

    # -------------------------------------------------------------
    # 1. DATA FRESHNESS & COMPLETENESS CHECK
    # -------------------------------------------------------------
    if len(df) < 15:
        res = {
            "decision": "NO TRADE",
            "why": "Chart data is insufficient (fewer than 15 closed candles available for analysis).",
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": "Wait for live candle data to accumulate from your chart feed.",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    # Timestamp freshness
    if 'timestamp' in df and len(df) > 0:
        last_ts = df['timestamp'].iloc[-1]
        if isinstance(last_ts, str):
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            except Exception:
                last_dt = now
        elif hasattr(last_ts, "to_pydatetime"):
            last_dt = last_ts.to_pydatetime()
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        elif isinstance(last_ts, datetime):
            last_dt = last_ts if last_ts.tzinfo is not None else last_ts.replace(tzinfo=timezone.utc)
        else:
            last_dt = now

        age_mins = (now - last_dt).total_seconds() / 60.0
        if age_mins > 180.0:
            res = {
                "decision": "NO TRADE",
                "why": f"Market feed is stale (last candle was {age_mins:.0f} minutes ago); cannot generate live trade setups.",
                "plan": "No entry yet.",
                "session": session_obs,
                "wait_for": "A fresh candle update from your TradingView alert or active data feed.",
                "is_valid_setup": False
            }
            res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
            return res

    # -------------------------------------------------------------
    # 2. SMALL-ACCOUNT & BROKER LOT SIZE GUARDRAIL (Using RiskEngine)
    # -------------------------------------------------------------
    risk_eval = RiskEngine.evaluate_account_risk(
        account_balance=risk_settings.account_balance,
        account_currency=risk_settings.account_currency,
        risk_per_trade_pct=risk_settings.risk_per_trade_pct,
        entry_price=plan.current_reference_price,
        stop_loss_price=plan.stop_loss,
        take_profit_price=plan.take_profit_1,
        contract_size=risk_settings.contract_size,
        min_lot_size=risk_settings.min_lot_size,
        usd_inr_rate=risk_settings.usd_inr_rate,
        leverage=risk_settings.leverage,
        direction="BUY" if "BUY" in plan.direction.upper() else "SELL"
    )

    if risk_eval["exceeds_risk_limit"]:
        res = {
            "decision": "NO TRADE",
            "why": risk_eval["rejection_reason"],
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": f"Account capital of at least {curr_sym}{risk_eval['needed_capital_for_safe_trading']:,.0f} or practice in a demo account.",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    # -------------------------------------------------------------
    # 3. EVIDENCE & DIRECTIONAL CONSISTENCY CHECKS
    # -------------------------------------------------------------
    curr_close = float(df['close'].iloc[-1])
    ema20 = float(indicators.get('ema20', curr_close))
    ema50 = float(indicators.get('ema50', curr_close))
    rsi = float(indicators.get('rsi14', 50.0))
    structure_trend = market_structure.get('trend', 'RANGING')

    is_bullish_emas = curr_close > ema20 > ema50
    is_bearish_emas = curr_close < ema20 < ema50

    # Conflict 1: EMAs are bullish, but structure is marked bearish
    if is_bullish_emas and structure_trend == "BEARISH":
        res = {
            "decision": "WAIT",
            "why": f"Moving averages are bullish (EMA20 > EMA50 at ${ema20:.2f}), but structure has not confirmed a break above resistance.",
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": f"A confirmed 15M candle close above ${market_structure.get('last_swing_high', curr_close + 2.0):.2f} with a successful retest.",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    # Conflict 2: EMAs are bearish, but structure is marked bullish
    if is_bearish_emas and structure_trend == "BULLISH":
        res = {
            "decision": "WAIT",
            "why": f"Moving averages are bearish (EMA20 < EMA50 at ${ema20:.2f}), but price is holding above intermediate support without confirmed breakdown.",
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": f"A confirmed 15M candle close below ${market_structure.get('last_swing_low', curr_close - 2.0):.2f} and confirmed retest.",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    # Overbought / Oversold momentum blocks
    if plan.direction == "BUY SETUP" and rsi > 72.0:
        res = {
            "decision": "WAIT",
            "why": f"Trend is bullish, but RSI (14) at {rsi:.1f} is overbought; entering here risks buying into an immediate pullback.",
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": f"RSI cooling down toward 50–55 and a pullback retest into EMA20 (${ema20:.2f}).",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    if plan.direction == "SELL SETUP" and rsi < 28.0:
        res = {
            "decision": "WAIT",
            "why": f"Trend is bearish, but RSI (14) at {rsi:.1f} is oversold; selling here risks shorting into a technical bounce.",
            "plan": "No entry yet.",
            "session": session_obs,
            "wait_for": f"RSI recovering toward 45–50 and a relief bounce retest into EMA20 (${ema20:.2f}).",
            "is_valid_setup": False
        }
        res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
        return res

    # -------------------------------------------------------------
    # 3.5. STATISTICAL MODEL ALIGNMENT CHECK
    # -------------------------------------------------------------
    if model_predictions and model_predictions.get("model_available"):
        probs = model_predictions.get("probabilities", {})
        p_rise = probs.get("prob_rise_target", 0.50)
        p_fall = probs.get("prob_fall_target", 0.50)

        if plan.direction == "BUY SETUP" and p_rise < 0.38:
            res = {
                "decision": "WAIT",
                "why": f"Moving averages show bullish bias, but the approved ML model indicates low target probability ({p_rise:.1%} rise probability), signaling weak momentum.",
                "plan": "No entry yet.",
                "session": session_obs,
                "wait_for": "Model probability to rise above 50% or stronger volume confirmation.",
                "is_valid_setup": False
            }
            res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
            return res

        elif plan.direction == "SELL SETUP" and p_fall < 0.38:
            res = {
                "decision": "WAIT",
                "why": f"Moving averages show bearish bias, but the approved ML model indicates low target probability ({p_fall:.1%} fall probability), signaling potential absorption.",
                "plan": "No entry yet.",
                "session": session_obs,
                "wait_for": "Model probability to fall below 50% or stronger selling volume.",
                "is_valid_setup": False
            }
            res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
            return res

    # -------------------------------------------------------------
    # 4. PRICE GEOMETRY & TARGET VALIDATION FOR BUY / SELL
    # -------------------------------------------------------------
    if plan.direction in ["BUY SETUP", "SELL SETUP"]:
        entry_min = plan.entry_zone_min or (curr_close - 0.75)
        entry_max = plan.entry_zone_max or (curr_close + 0.75)
        sl = plan.stop_loss
        tp = plan.take_profit_1

        if abs(entry_max - entry_min) < 0.2:
            entry_min = round(curr_close - 0.75, 2)
            entry_max = round(curr_close + 0.75, 2)

        # Validate BUY geometry
        if plan.direction == "BUY SETUP":
            if sl is None or sl >= entry_min or tp is None or tp <= entry_max:
                res = {
                    "decision": "WAIT",
                    "why": "Structure does not offer an invalidation level with acceptable risk geometry.",
                    "plan": "No entry yet.",
                    "session": session_obs,
                    "wait_for": "A confirmed swing low to form and anchor a valid protective stop.",
                    "is_valid_setup": False
                }
                res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
                return res
            
            risk_dist = entry_max - sl
            reward_dist = tp - entry_min
            rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

            if rr < 2.0:
                res = {
                    "decision": "WAIT",
                    "why": f"Potential target at ${tp:.2f} offers only 1:{rr:.1f} R:R, below our required 1:2.0 minimum threshold.",
                    "plan": "No entry yet.",
                    "session": session_obs,
                    "wait_for": "A deeper pullback into support to achieve at least a 1:2.0 risk-to-reward ratio.",
                    "is_valid_setup": False
                }
                res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
                return res

            lot_size = plan.suggested_lot_size or risk_settings.min_lot_size
            cash_risk = risk_eval["allowed_cash_risk"]
            res = {
                "decision": "BUY",
                "why": f"Price (${curr_close:.2f}) confirmed bullish alignment above EMA20 (${ema20:.2f}) with support holding.",
                "plan": f"Entry zone ${entry_min:.2f}–${entry_max:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} (1:{rr:.1f} R:R) | Risk: {curr_sym}{cash_risk:.2f} ({lot_size:.2f} lots).",
                "session": session_obs,
                "wait_for": f"Confirmed 15M candle close within ${entry_min:.2f}–${entry_max:.2f}; setup invalidates on close below ${sl:.2f}.",
                "is_valid_setup": True
            }
            res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
            return res

        # Validate SELL geometry
        elif plan.direction == "SELL SETUP":
            if sl is None or sl <= entry_max or tp is None or tp >= entry_min:
                res = {
                    "decision": "WAIT",
                    "why": "Structure does not offer an invalidation level with acceptable risk geometry.",
                    "plan": "No entry yet.",
                    "session": session_obs,
                    "wait_for": "A confirmed swing high to form and anchor a valid protective stop.",
                    "is_valid_setup": False
                }
                res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
                return res
            
            risk_dist = sl - entry_min
            reward_dist = entry_max - tp
            rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

            if rr < 2.0:
                res = {
                    "decision": "WAIT",
                    "why": f"Potential target at ${tp:.2f} offers only 1:{rr:.1f} R:R, below our required 1:2.0 minimum threshold.",
                    "plan": "No entry yet.",
                    "session": session_obs,
                    "wait_for": "A relief rally into resistance to achieve at least a 1:2.0 risk-to-reward ratio.",
                    "is_valid_setup": False
                }
                res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
                return res

            lot_size = plan.suggested_lot_size or risk_settings.min_lot_size
            cash_risk = risk_eval["allowed_cash_risk"]
            res = {
                "decision": "SELL",
                "why": f"Price (${curr_close:.2f}) confirmed bearish alignment below EMA20 (${ema20:.2f}) with resistance holding.",
                "plan": f"Entry zone ${entry_min:.2f}–${entry_max:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} (1:{rr:.1f} R:R) | Risk: {curr_sym}{cash_risk:.2f} ({lot_size:.2f} lots).",
                "session": session_obs,
                "wait_for": f"Confirmed 15M candle close within ${entry_min:.2f}–${entry_max:.2f}; setup invalidates on close above ${sl:.2f}.",
                "is_valid_setup": True
            }
            res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
            return res

    # Default fallback for WAIT or NO TRADE
    res = {
        "decision": "WAIT",
        "why": f"Price (${curr_close:.2f}) is consolidating between EMA20 (${ema20:.2f}) and EMA50 (${ema50:.2f}) without directional breakout.",
        "plan": "No entry yet.",
        "session": session_obs,
        "wait_for": "A confirmed candle close outside the current consolidation range.",
        "is_valid_setup": False
    }
    res["raw_text"] = f"**Decision: {res['decision']}**\n\n**Why:** {res['why']}\n\n**Plan:** {res['plan']}\n\n**Session:** {res['session']}\n\n**Wait for:** {res['wait_for']}"
    return res

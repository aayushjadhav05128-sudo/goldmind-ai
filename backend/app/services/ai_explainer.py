import httpx
from typing import Optional, Dict, Any
from app.config import settings
from app.schemas.signal import TradePlan

async def generate_ai_explanation(
    trade_plan: TradePlan,
    indicators: Dict[str, Any],
    market_structure: Dict[str, Any]
) -> str:
    """
    Generates a structured, evidence-grounded technical narrative.
    Strictly quotes calculated indicators, price levels, and timestamps.
    Never hallucinates prices or overrides calculated risk parameters.
    """
    # Deterministic grounded baseline template (always accurate and reliable)
    evidence_bullets = "\n".join([f"• {e}" for e in trade_plan.technical_evidence]) if trade_plan.technical_evidence else "• Structure is currently consolidating."
    counter_bullets = "\n".join([f"• {c}" for c in trade_plan.counter_evidence]) if trade_plan.counter_evidence else "• No critical structural counter-evidence detected."
    inval_bullets = "\n".join([f"• {i}" for i in trade_plan.invalidation_conditions]) if trade_plan.invalidation_conditions else "• Breach of current session extremes."
    wait_bullets = "\n".join([f"• {w}" for w in trade_plan.wait_conditions]) if trade_plan.wait_conditions else "• Enter on candle close confirmation."

    sizing_note = ""
    if not trade_plan.trade_permitted and trade_plan.sizing_warning:
        sizing_note = f"\n⚠️ CAPITAL PROTECTION ALERT:\n{trade_plan.sizing_warning}\n"

    deterministic_summary = f"""### {trade_plan.direction} — {trade_plan.setup_status}

**Symbol:** {trade_plan.symbol} | **Timeframe:** {trade_plan.timeframe} | **Ref Price:** {trade_plan.current_reference_price:.2f}
**Timestamp:** {trade_plan.source_timestamp} | **Session:** {indicators.get('session', 'ACTIVE')}

#### Market Context & Structural Evidence:
{evidence_bullets}

#### Risk & Level Specifications:
• **Proposed Entry Zone:** {trade_plan.entry_zone_min:.2f} – {trade_plan.entry_zone_max:.2f} (Mid: {trade_plan.proposed_entry:.2f})
• **Calculated Stop Loss:** {trade_plan.stop_loss:.2f}
• **Targets:** TP1: {trade_plan.take_profit_1:.2f} (R:R {trade_plan.rr_tp1:.1f}), TP2: {trade_plan.take_profit_2:.2f} (R:R {trade_plan.rr_tp2:.1f}), TP3: {trade_plan.take_profit_3:.2f} (R:R {trade_plan.rr_tp3:.1f})
• **Capital Sizing:** Suggested {trade_plan.suggested_lot_size:.2f} lots | Permitted Risk: {trade_plan.account_currency} {trade_plan.max_cash_risk:.2f} ({trade_plan.risk_percentage}%)
{sizing_note}
#### Counter-Evidence & Risk Factors:
{counter_bullets}

#### Entry & Confirmation Requirements:
{wait_bullets}

#### Invalidation Conditions:
{inval_bullets}

*Notice: This trade plan is an analytical aid based strictly on confirmed technical features. It does not constitute financial advice or guarantee profitability.*"""

    # If Gemini API Key is configured and provider is gemini, request enhanced synthesis
    if settings.AI_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY:
        try:
            prompt = f"""You are GoldMind AI quantitative forex analyst.
Based ONLY on these calculated facts (do not invent prices or dates):
Symbol: {trade_plan.symbol}
Direction: {trade_plan.direction}
Ref Price: {trade_plan.current_reference_price}
Stop Loss: {trade_plan.stop_loss}
Take Profit 1: {trade_plan.take_profit_1}
RSI: {indicators.get('rsi14')}
EMA20: {indicators.get('ema20')}, EMA50: {indicators.get('ema50')}, EMA200: {indicators.get('ema200')}
Session: {indicators.get('session')}
Evidence: {trade_plan.technical_evidence}
Counter-evidence: {trade_plan.counter_evidence}
Capital Note: {trade_plan.sizing_warning}

Write a professional, concise, objective 3-paragraph institutional trade briefing adhering to this exact state. Never claim 100% certainty or guaranteed win rate."""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=6.0) as client:
                res = await client.post(url, json={
                    "contents": [{"parts": [{"text": prompt}]}]
                })
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        llm_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if llm_text:
                            return f"{llm_text}\n\n---\n{deterministic_summary}"
        except Exception:
            # Safely fall back to deterministic summary on any network or API issue
            pass

    return deterministic_summary

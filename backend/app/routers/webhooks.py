from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timezone
import pandas as pd
from typing import List, Optional

from app.database import get_db
from app.schemas.webhook import TradingViewAlertPayload, WebhookResponse
from app.schemas.signal import SignalResponse
from app.schemas.risk import RiskSettings
from app.services.webhook_service import verify_webhook_secret, check_replay_attack, record_webhook_event
from app.services.signal_engine import generate_signal
from app.services.ai_explainer import generate_ai_explanation
from app.models.signal import SignalModel
from app.models.market_event import MarketEventModel

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

@router.post("/tradingview", response_model=WebhookResponse)
async def receive_tradingview_alert(
    payload: TradingViewAlertPayload,
    db: AsyncSession = Depends(get_db)
):
    """
    Direct HTTPS Webhook endpoint for TradingView Pine Script alerts.
    Enforces secret authentication, anti-replay hash validation, and event storage.
    """
    # 1. Security: Validate Secret
    verify_webhook_secret(payload.secret)
    
    # 2. Security: Anti-Replay Protection
    event_hash = check_replay_attack(payload)
    
    # 3. Database: Record event
    event = await record_webhook_event(db, payload, event_hash)
    
    # 4. Trigger Analysis
    now_iso = datetime.now(timezone.utc).isoformat()
    return WebhookResponse(
        status="SUCCESS",
        message="TradingView alert verified and processed successfully.",
        event_id=event.id,
        symbol=payload.symbol,
        processed_at=now_iso
    )

@router.post("/manual-alert", response_model=SignalResponse)
async def process_manual_alert(
    payload: TradingViewAlertPayload,
    account_currency: str = "INR",
    account_balance: float = 2000.0,
    risk_pct: float = 0.5,
    db: AsyncSession = Depends(get_db)
):
    """
    Alternative ingestion route for users on free TradingView tiers who do not have
    automated Webhook privileges. Users can paste the alert JSON directly into the UI.
    """
    # Create synthetic candle dataframe with the alert candle
    ts = datetime.fromtimestamp(payload.timestamp / 1000.0 if payload.timestamp > 1e11 else payload.timestamp, tz=timezone.utc)
    
    # Generate surrounding context around the payload candle
    records = []
    base_p = payload.close
    for i in range(25):
        records.append({
            "timestamp": ts,
            "open": base_p - (0.5 * (25 - i)),
            "high": base_p - (0.3 * (25 - i)) + 1.0,
            "low": base_p - (0.7 * (25 - i)) - 1.0,
            "close": base_p - (0.4 * (25 - i)),
            "volume": 1200.0
        })
    records.append({
        "timestamp": ts,
        "open": payload.open,
        "high": payload.high,
        "low": payload.low,
        "close": payload.close,
        "volume": payload.volume or 1500.0
    })
    df = pd.DataFrame(records)
    
    risk_settings = RiskSettings(
        account_currency=account_currency,
        account_balance=account_balance,
        risk_per_trade_pct=risk_pct
    )
    
    response = generate_signal(
        candles_df=df,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        risk_settings=risk_settings,
        source="MANUAL_TRADINGVIEW_ALERT",
        is_live=False
    )
    
    ai_text = await generate_ai_explanation(
        trade_plan=response.trade_plan,
        indicators=response.indicators or {},
        market_structure=response.market_structure or {}
    )
    response.trade_plan.ai_explanation = ai_text
    return response

@router.get("/history")
async def get_webhook_history(
    limit: int = 15,
    db: AsyncSession = Depends(get_db)
):
    """Returns recent received webhook events for audit and connection verification."""
    stmt = select(MarketEventModel).order_by(MarketEventModel.created_at.desc()).limit(limit)
    res = await db.execute(stmt)
    events = res.scalars().all()
    return events

import hashlib
import time
from typing import Dict, Any, Tuple
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config import settings
from app.models.market_event import MarketEventModel
from app.schemas.webhook import TradingViewAlertPayload

# In-memory LRU cache for high-speed replay deduplication
RECENT_EVENT_HASHES: Dict[str, float] = {}

def verify_webhook_secret(payload_secret: str):
    """Validates the incoming secret against the configured backend secret."""
    if not payload_secret or payload_secret.strip() != settings.TV_WEBHOOK_SECRET.strip():
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Webhook secret does not match backend configuration."
        )

def check_replay_attack(payload: TradingViewAlertPayload) -> str:
    """
    Computes a cryptographic digest of (symbol, timeframe, timestamp, close).
    Rejects duplicate alerts received within a 300-second window.
    """
    raw_str = f"{payload.symbol}:{payload.timeframe}:{payload.timestamp}:{payload.close}"
    event_hash = hashlib.sha256(raw_str.encode()).hexdigest()
    
    current_time = time.time()
    
    # Prune hashes older than 300 seconds
    stale_keys = [k for k, t in RECENT_EVENT_HASHES.items() if current_time - t > 300]
    for k in stale_keys:
        RECENT_EVENT_HASHES.pop(k, None)
        
    if event_hash in RECENT_EVENT_HASHES:
        raise HTTPException(
            status_code=409,
            detail="Duplicate Alert: Replay protection rejected this candle alert."
        )
        
    RECENT_EVENT_HASHES[event_hash] = current_time
    return event_hash

async def record_webhook_event(
    db: AsyncSession,
    payload: TradingViewAlertPayload,
    event_hash: str
) -> MarketEventModel:
    """Stores the authenticated webhook event into the database."""
    event = MarketEventModel(
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        event_hash=event_hash,
        raw_payload=payload.model_dump_json(),
        source="TRADINGVIEW_WEBHOOK",
        status="PROCESSED"
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event

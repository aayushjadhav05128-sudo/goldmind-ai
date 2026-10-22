from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from app.database import get_db
from app.models.signal import SignalModel

router = APIRouter(prefix="/signals", tags=["Signals"])

@router.get("/")
async def list_signals(
    limit: int = Query(30, ge=1, le=100),
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Lists historically issued signals with filtering."""
    query = select(SignalModel).order_by(SignalModel.created_at.desc())
    if symbol:
        query = query.where(SignalModel.symbol == symbol.upper())
    if direction:
        query = query.where(SignalModel.direction == direction)
    query = query.limit(limit)
    res = await db.execute(query)
    signals = res.scalars().all()
    return signals

@router.get("/{signal_id}")
async def get_signal_detail(
    signal_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Retrieves single signal by ID."""
    stmt = select(SignalModel).where(SignalModel.id == signal_id)
    res = await db.execute(stmt)
    sig = res.scalar_one_or_none()
    if not sig:
        raise HTTPException(status_code=404, detail="Signal not found.")
    return sig

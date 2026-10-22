from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from datetime import datetime, timezone

from app.database import get_db
from app.models.journal import JournalEntryModel
from app.schemas.journal import JournalEntryCreate, JournalEntryUpdate, JournalEntryResponse

router = APIRouter(prefix="/journal", tags=["Trade Journal"])

@router.get("/", response_model=List[JournalEntryResponse])
async def list_journal_entries(
    db: AsyncSession = Depends(get_db)
):
    """Lists all user trade journal entries."""
    stmt = select(JournalEntryModel).order_by(JournalEntryModel.executed_at.desc())
    res = await db.execute(stmt)
    entries = res.scalars().all()
    return entries

@router.post("/", response_model=JournalEntryResponse)
async def create_journal_entry(
    entry: JournalEntryCreate,
    db: AsyncSession = Depends(get_db)
):
    """Saves a trade setup to the journal before or after manual broker execution."""
    new_entry = JournalEntryModel(
        signal_id=entry.signal_id,
        symbol=entry.symbol,
        direction=entry.direction,
        planned_entry=entry.planned_entry,
        planned_sl=entry.planned_sl,
        planned_tp=entry.planned_tp,
        actual_entry=entry.actual_entry or entry.planned_entry,
        lot_size=entry.lot_size,
        account_currency=entry.account_currency,
        notes=entry.notes,
        chart_url=entry.chart_url,
        outcome="OPEN"
    )
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry

@router.put("/{entry_id}", response_model=JournalEntryResponse)
async def update_journal_entry(
    entry_id: int,
    update_data: JournalEntryUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Updates trade outcome, exit price, realized P&L and R-multiples."""
    stmt = select(JournalEntryModel).where(JournalEntryModel.id == entry_id)
    res = await db.execute(stmt)
    entry = res.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found.")
        
    if update_data.actual_exit is not None:
        entry.actual_exit = update_data.actual_exit
        entry.closed_at = datetime.now(timezone.utc)
    if update_data.realized_pnl is not None:
        entry.realized_pnl = update_data.realized_pnl
    if update_data.r_multiple is not None:
        entry.r_multiple = update_data.r_multiple
    if update_data.outcome is not None:
        entry.outcome = update_data.outcome
    if update_data.notes is not None:
        entry.notes = update_data.notes
    if update_data.chart_url is not None:
        entry.chart_url = update_data.chart_url
        
    await db.commit()
    await db.refresh(entry)
    return entry

@router.get("/stats")
async def get_journal_stats(db: AsyncSession = Depends(get_db)):
    """Computes real verified performance stats from closed journal trades."""
    stmt = select(JournalEntryModel)
    res = await db.execute(stmt)
    entries = res.scalars().all()
    
    closed = [e for e in entries if e.outcome in ["WIN", "LOSS", "BREAKEVEN"]]
    if not closed:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_r": 0.0,
            "open_trades": len([e for e in entries if e.outcome == "OPEN"])
        }
        
    wins = [e for e in closed if e.outcome == "WIN"]
    losses = [e for e in closed if e.outcome == "LOSS"]
    win_rate = round((len(wins) / len(closed)) * 100.0, 1)
    total_pnl = round(sum([e.realized_pnl for e in closed]), 2)
    avg_r = round(sum([e.r_multiple for e in closed]) / len(closed), 2)
    
    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_r": avg_r,
        "open_trades": len([e for e in entries if e.outcome == "OPEN"])
    }

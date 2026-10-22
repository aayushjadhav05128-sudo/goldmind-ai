from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class JournalEntryCreate(BaseModel):
    signal_id: Optional[int] = None
    symbol: str = "XAUUSD"
    direction: str = "BUY" # BUY or SELL
    planned_entry: float
    planned_sl: float
    planned_tp: float
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    lot_size: float = 0.01
    account_currency: str = "INR"
    notes: Optional[str] = None
    chart_url: Optional[str] = None

class JournalEntryUpdate(BaseModel):
    actual_exit: Optional[float] = None
    realized_pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    outcome: Optional[str] = None # WIN, LOSS, BREAKEVEN, OPEN
    notes: Optional[str] = None
    chart_url: Optional[str] = None

class JournalEntryResponse(BaseModel):
    id: int
    signal_id: Optional[int] = None
    symbol: str
    direction: str
    planned_entry: float
    planned_sl: float
    planned_tp: float
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    lot_size: float
    account_currency: str
    realized_pnl: float
    r_multiple: float
    outcome: str
    notes: Optional[str] = None
    chart_url: Optional[str] = None
    executed_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

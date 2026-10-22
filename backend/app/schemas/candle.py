from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0

class CandleHistory(BaseModel):
    symbol: str
    timeframe: str
    source: str # "LIVE_FEED", "YAHOO", "CSV", "SIMULATED", "MANUAL"
    last_updated: datetime
    is_live: bool = False
    is_stale: bool = False
    candles: List[Candle]

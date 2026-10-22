from pydantic import BaseModel, Field
from typing import Optional

class TradingViewAlertPayload(BaseModel):
    secret: str = Field(..., description="Webhook authentication secret")
    symbol: str = Field(default="XAUUSD", description="Symbol ticker e.g. XAUUSD")
    timeframe: str = Field(default="15", description="Chart timeframe period")
    timestamp: float = Field(..., description="Candle timestamp (UNIX epoch ms or sec)")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    structure: Optional[str] = None
    condition: Optional[str] = None

class WebhookResponse(BaseModel):
    status: str
    message: str
    event_id: Optional[int] = None
    generated_signal_id: Optional[int] = None
    symbol: str
    processed_at: str

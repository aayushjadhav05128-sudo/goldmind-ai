from sqlalchemy import Column, Integer, String, Float, DateTime, Text, func
from app.database import Base

class MarketEventModel(Base):
    __tablename__ = "market_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))
    event_hash = Column(String(64), unique=True, index=True) # for replay protection
    raw_payload = Column(Text, nullable=False)
    source = Column(String(50), default="TRADINGVIEW_WEBHOOK")
    status = Column(String(30), default="PROCESSED")
    created_at = Column(DateTime, default=func.now())

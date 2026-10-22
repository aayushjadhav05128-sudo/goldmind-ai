from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, func
from app.database import Base

class JournalEntryModel(Base):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    symbol = Column(String(20), default="XAUUSD")
    direction = Column(String(20), nullable=False) # BUY or SELL
    
    # Planned vs Actual
    planned_entry = Column(Float, nullable=False)
    planned_sl = Column(Float, nullable=False)
    planned_tp = Column(Float, nullable=False)
    actual_entry = Column(Float, nullable=True)
    actual_exit = Column(Float, nullable=True)
    
    lot_size = Column(Float, default=0.01)
    account_currency = Column(String(10), default="INR")
    
    realized_pnl = Column(Float, default=0.0)
    realized_pnl_usd = Column(Float, default=0.0)
    r_multiple = Column(Float, default=0.0)
    outcome = Column(String(20), default="OPEN") # OPEN, WIN, LOSS, BREAKEVEN
    
    notes = Column(Text, nullable=True)
    chart_url = Column(String(500), nullable=True)
    
    executed_at = Column(DateTime, default=func.now())
    closed_at = Column(DateTime, nullable=True)

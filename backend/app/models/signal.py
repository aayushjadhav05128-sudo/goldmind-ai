from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, func
from app.database import Base
import datetime

class SignalModel(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String(20), index=True, default="XAUUSD")
    timeframe = Column(String(10), default="15m")
    direction = Column(String(30), nullable=False)  # BUY SETUP, SELL SETUP, WAIT FOR CONFIRMATION, NO TRADE
    status = Column(String(30), default="CONFIRMED") # CONFIRMED, PENDING_CONFIRMATION, INVALIDATED
    
    # Price levels
    reference_price = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    entry_zone_low = Column(Float, nullable=True)
    entry_zone_high = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit_1 = Column(Float, nullable=True)
    take_profit_2 = Column(Float, nullable=True)
    take_profit_3 = Column(Float, nullable=True)
    
    # R:R Metrics
    risk_reward_tp1 = Column(Float, nullable=True)
    risk_reward_tp2 = Column(Float, nullable=True)
    risk_reward_tp3 = Column(Float, nullable=True)
    
    # Scoring & Confidence (0-100)
    score = Column(Float, default=0.0)
    rule_alignment_score = Column(Float, default=0.0)
    data_quality_score = Column(Float, default=100.0)
    
    # Sizing & Capital
    account_currency = Column(String(10), default="INR")
    account_balance = Column(Float, default=2000.0)
    risk_percentage = Column(Float, default=0.5)
    max_cash_risk = Column(Float, default=10.0) # in account currency
    suggested_lot_size = Column(Float, default=0.0)
    sizing_skipped_reason = Column(String(255), nullable=True)
    
    # Detailed Context
    technical_evidence = Column(Text, nullable=True) # JSON string
    counter_evidence = Column(Text, nullable=True)   # JSON string
    invalidation_conditions = Column(Text, nullable=True) # JSON string
    wait_conditions = Column(Text, nullable=True)         # JSON string
    
    # AI Explanation
    ai_explanation = Column(Text, nullable=True)
    
    # Metadata
    source = Column(String(50), default="LIVE_ANALYSIS") # WEBHOOK, LIVE_ANALYSIS, MANUAL, BACKTEST
    source_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

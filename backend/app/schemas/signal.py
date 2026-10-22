from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TradePlan(BaseModel):
    id: Optional[int] = None
    symbol: str = "XAUUSD"
    direction: str = "WAIT FOR CONFIRMATION" # "BUY SETUP", "SELL SETUP", "WAIT FOR CONFIRMATION", "NO TRADE"
    setup_status: str = "PENDING_CONFIRMATION" # "CONFIRMED", "PENDING_CONFIRMATION", "INVALIDATED"
    timeframe: str = "15m"
    
    current_reference_price: float
    source_timestamp: str
    data_source: str = "LIVE_DATA"
    is_live_data: bool = True
    
    # Entry & Targets
    proposed_entry: Optional[float] = None
    entry_zone_min: Optional[float] = None
    entry_zone_max: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    # Risk-to-Reward
    rr_tp1: Optional[float] = None
    rr_tp2: Optional[float] = None
    rr_tp3: Optional[float] = None
    
    # Position Sizing & Capital
    account_currency: str = "INR"
    account_balance: float = 2000.0
    risk_percentage: float = 0.5
    max_cash_risk: float = 10.0
    suggested_lot_size: float = 0.0
    trade_permitted: bool = False
    sizing_warning: Optional[str] = None
    recommend_demo_mode: bool = False
    
    # Qualitative / Grounded Analysis
    technical_evidence: List[str] = []
    counter_evidence: List[str] = []
    invalidation_conditions: List[str] = []
    wait_conditions: List[str] = []
    holding_period: str = "Intraday (2h - 8h)"
    signal_expiry: str = "Next Higher Timeframe Candle Close"
    
    # AI Grounded Explanation
    ai_explanation: Optional[str] = None
    
    # Scores
    rule_score: float = 0.0 # 0 - 100
    data_quality_score: float = 100.0 # 0 - 100

class SignalResponse(BaseModel):
    status: str
    trade_plan: TradePlan
    market_structure: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, Any]] = None
    multi_timeframe: Optional[Dict[str, Any]] = None
    model_predictions: Optional[Dict[str, Any]] = None

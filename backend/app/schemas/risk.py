from pydantic import BaseModel, Field
from typing import Optional

class RiskSettings(BaseModel):
    account_currency: str = Field(default="INR", description="INR or USD")
    account_balance: float = Field(default=2000.0, ge=100.0, description="Account balance")
    risk_per_trade_pct: float = Field(default=0.5, ge=0.1, le=5.0, description="Risk per trade percentage")
    max_risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0, description="Max allowed risk per trade")
    max_daily_loss_pct: float = Field(default=2.0, ge=0.5, le=10.0, description="Max daily loss limit")
    max_concurrent_trades: int = Field(default=1, ge=1, le=5, description="Max concurrent open trades")
    
    usd_inr_rate: float = Field(default=86.50, description="INR per USD conversion rate")
    contract_size: float = Field(default=100.0, description="Contract size (100 oz standard, 10 mini, 1 micro)")
    min_lot_size: float = Field(default=0.01, description="Broker minimum lot size")
    lot_step: float = Field(default=0.01, description="Broker lot increment")
    leverage: float = Field(default=100.0, description="Broker leverage ratio e.g. 100 for 1:100")
    spread_points: float = Field(default=0.25, description="Estimated spread in points ($0.25 = 2.5 pips)")
    commission_per_lot: float = Field(default=7.0, description="Commission per round lot in USD")

class PositionSizeResult(BaseModel):
    cash_risk_permitted_account_currency: float
    cash_risk_usd: float
    entry_price: float
    stop_loss_price: float
    sl_distance_points: float
    
    theoretical_lot_size: float
    suggested_lot_size: float
    trade_permitted: bool
    warning_message: Optional[str] = None
    recommend_demo_mode: bool = False
    
    estimated_margin_required_usd: float
    estimated_margin_required_account_currency: float
    estimated_loss_at_sl_account_currency: float
    estimated_profit_tp1_account_currency: float
    estimated_profit_tp2_account_currency: float
    estimated_profit_tp3_account_currency: float
    
    cost_adjusted_rr_tp1: float
    cost_adjusted_rr_tp2: float
    cost_adjusted_rr_tp3: float

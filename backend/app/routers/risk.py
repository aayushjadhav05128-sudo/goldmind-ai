from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.schemas.risk import RiskSettings, PositionSizeResult
from app.services.risk_manager import calculate_position_size
from app.config import settings

router = APIRouter(prefix="/risk", tags=["Risk Management"])

class PositionCalculationRequest(BaseModel):
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    settings: Optional[RiskSettings] = None

@router.post("/calculate-position", response_model=PositionSizeResult)
def calculate_position(req: PositionCalculationRequest):
    """
    Calculates exact lot size, cash risk in INR/USD, margin requirements,
    and returns trade permission status with the crucial small-account guardrails.
    """
    res = calculate_position_size(
        entry_price=req.entry_price,
        stop_loss_price=req.stop_loss,
        take_profit_1=req.take_profit_1,
        take_profit_2=req.take_profit_2,
        take_profit_3=req.take_profit_3,
        settings=req.settings
    )
    return res

@router.get("/defaults", response_model=RiskSettings)
def get_default_risk_settings():
    """Returns application default risk settings."""
    return RiskSettings(
        account_currency=settings.DEFAULT_ACCOUNT_CURRENCY,
        account_balance=settings.DEFAULT_ACCOUNT_BALANCE,
        risk_per_trade_pct=settings.DEFAULT_RISK_PER_TRADE_PCT,
        max_risk_per_trade_pct=settings.MAX_RISK_PER_TRADE_PCT,
        max_daily_loss_pct=settings.MAX_DAILY_LOSS_PCT,
        max_concurrent_trades=settings.MAX_CONCURRENT_TRADES,
        usd_inr_rate=settings.USD_INR_RATE,
        contract_size=settings.XAUUSD_CONTRACT_SIZE,
        min_lot_size=settings.BROKER_MIN_LOT,
        lot_step=settings.BROKER_LOT_STEP,
        leverage=settings.DEFAULT_LEVERAGE
    )

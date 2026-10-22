from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from app.database import get_db
from app.schemas.signal import SignalResponse
from app.schemas.risk import RiskSettings
from app.services.market_data.factory import market_data_factory
from app.services.signal_engine import generate_signal
from app.services.ai_explainer import generate_ai_explanation
from app.models.signal import SignalModel

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.get("/latest", response_model=SignalResponse)
async def get_latest_analysis(
    symbol: str = Query("XAUUSD", description="Symbol to analyze"),
    timeframe: str = Query("15m", description="Primary timeframe"),
    account_currency: str = Query("INR", description="INR or USD"),
    account_balance: float = Query(2000.0, description="Account balance"),
    risk_pct: float = Query(0.5, description="Risk per trade %"),
    db: AsyncSession = Depends(get_db)
):
    """
    Fetches latest candles from active market data provider, executes multi-timeframe
    market structure and price action calculations, and returns a verified trade plan.
    """
    provider = market_data_factory.get_provider()
    history = await provider.fetch_candles(symbol=symbol, timeframe=timeframe, limit=120)
    
    # Convert candles to DataFrame
    records = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        }
        for c in history.candles
    ]
    df = pd.DataFrame(records)
    
    risk_settings = RiskSettings(
        account_currency=account_currency,
        account_balance=account_balance,
        risk_per_trade_pct=risk_pct
    )
    
    response = generate_signal(
        candles_df=df,
        symbol=symbol,
        timeframe=timeframe,
        risk_settings=risk_settings,
        source=history.source,
        is_live=history.is_live
    )
    
    # Machine Learning point-in-time inference
    from app.services.ml.predictor import LivePredictor
    predictor = LivePredictor(target_label="label_triple_barrier_buy")
    model_predictions = predictor.predict_current_state(df, timeframe=timeframe)
    response.model_predictions = model_predictions

    # Apply Deterministic Validation Layer
    from app.services.validation_layer import validate_and_finalize_decision
    validation = validate_and_finalize_decision(
        plan=response.trade_plan,
        df=df,
        indicators=response.indicators or {},
        market_structure=response.market_structure or {},
        risk_settings=risk_settings,
        model_predictions=model_predictions
    )
    
    response.trade_plan.direction = validation["decision"]
    response.trade_plan.ai_explanation = validation["raw_text"]
    
    if validation["decision"] in ["WAIT", "NO TRADE"]:
        response.trade_plan.proposed_entry = None
        response.trade_plan.entry_zone_min = None
        response.trade_plan.entry_zone_max = None
        response.trade_plan.stop_loss = None
        response.trade_plan.take_profit_1 = None
        response.trade_plan.take_profit_2 = None
        response.trade_plan.take_profit_3 = None
        response.trade_plan.setup_status = "PENDING_CONFIRMATION" if validation["decision"] == "WAIT" else "INVALIDATED"
    else:
        response.trade_plan.setup_status = "CONFIRMED"
    
    # Save signal in database for audit trail if it's a valid setup
    if response.trade_plan.direction in ["BUY SETUP", "SELL SETUP", "WAIT FOR CONFIRMATION"]:
        db_signal = SignalModel(
            symbol=symbol,
            timeframe=timeframe,
            direction=response.trade_plan.direction,
            status=response.trade_plan.setup_status,
            reference_price=response.trade_plan.current_reference_price,
            entry_price=response.trade_plan.proposed_entry,
            entry_zone_low=response.trade_plan.entry_zone_min,
            entry_zone_high=response.trade_plan.entry_zone_max,
            stop_loss=response.trade_plan.stop_loss,
            take_profit_1=response.trade_plan.take_profit_1,
            take_profit_2=response.trade_plan.take_profit_2,
            take_profit_3=response.trade_plan.take_profit_3,
            risk_reward_tp1=response.trade_plan.rr_tp1,
            risk_reward_tp2=response.trade_plan.rr_tp2,
            risk_reward_tp3=response.trade_plan.rr_tp3,
            score=response.trade_plan.rule_score,
            account_currency=risk_settings.account_currency,
            account_balance=risk_settings.account_balance,
            risk_percentage=risk_settings.risk_per_trade_pct,
            max_cash_risk=response.trade_plan.max_cash_risk,
            suggested_lot_size=response.trade_plan.suggested_lot_size,
            sizing_skipped_reason=response.trade_plan.sizing_warning,
            technical_evidence="; ".join(response.trade_plan.technical_evidence),
            counter_evidence="; ".join(response.trade_plan.counter_evidence),
            invalidation_conditions="; ".join(response.trade_plan.invalidation_conditions),
            wait_conditions="; ".join(response.trade_plan.wait_conditions),
            ai_explanation=ai_text,
            source=history.source
        )
        db.add(db_signal)
        await db.commit()
        await db.refresh(db_signal)
        response.trade_plan.id = db_signal.id

    return response

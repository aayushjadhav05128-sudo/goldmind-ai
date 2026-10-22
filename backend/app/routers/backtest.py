from fastapi import APIRouter, Query, UploadFile, File
import pandas as pd
import io
from typing import Optional
from app.services.market_data.factory import market_data_factory
from app.services.backtester import run_backtest

router = APIRouter(prefix="/backtest", tags=["Backtesting"])

@router.post("/run")
async def execute_backtest(
    symbol: str = Query("XAUUSD"),
    timeframe: str = Query("15m"),
    spread_points: float = Query(0.25, description="Spread in points ($0.25)"),
    slippage_points: float = Query(0.10, description="Slippage in points"),
    lot_size: float = Query(0.01, description="Lot size")
):
    """
    Executes walk-forward backtest using available candles from active provider.
    Enforces zero lookahead bias and conservative same-candle execution.
    """
    provider = market_data_factory.get_provider()
    history = await provider.fetch_candles(symbol=symbol, timeframe=timeframe, limit=200)
    
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
    
    res = run_backtest(
        df=df,
        spread_points=spread_points,
        slippage_points=slippage_points,
        lot_size=lot_size
    )
    return res

@router.post("/upload-csv")
async def backtest_uploaded_csv(
    file: UploadFile = File(...)
):
    """Backtests a user-uploaded CSV file directly."""
    contents = await file.read()
    csv_str = contents.decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_str))
    
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)
    
    res = run_backtest(df=df)
    return res

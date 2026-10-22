from fastapi import APIRouter, Query, UploadFile, File
from app.services.market_data.factory import market_data_factory
from app.schemas.candle import CandleHistory

router = APIRouter(prefix="/market-data", tags=["Market Data"])

@router.get("/candles", response_model=CandleHistory)
async def get_candles(
    symbol: str = Query("XAUUSD"),
    timeframe: str = Query("15m"),
    limit: int = Query(100, ge=10, le=500)
):
    """Fetches normalized OHLCV candles from the active provider."""
    provider = market_data_factory.get_provider()
    history = await provider.fetch_candles(symbol=symbol, timeframe=timeframe, limit=limit)
    return history

@router.get("/status")
def get_provider_status():
    """Returns active data feed status, latency, freshness, and source."""
    provider = market_data_factory.get_provider()
    return provider.get_provider_status()

@router.post("/upload-csv")
async def upload_candles_csv(file: UploadFile = File(...)):
    """Uploads and activates a custom historical CSV candle dataset."""
    contents = await file.read()
    csv_text = contents.decode("utf-8")
    market_data_factory.load_csv(csv_text)
    return {
        "status": "SUCCESS",
        "message": f"Successfully loaded CSV '{file.filename}'. Market feed switched to CSV historical mode."
    }

@router.post("/reset-simulated")
def reset_to_simulated():
    """Resets the active market data feed to the simulated gold price engine."""
    market_data_factory.set_simulated()
    return {"status": "SUCCESS", "message": "Market feed reset to Simulated Market Engine."}

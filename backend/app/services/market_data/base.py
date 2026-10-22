from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from app.schemas.candle import CandleHistory

class MarketDataProvider(ABC):
    """Abstract interface for all Forex & Gold market data providers."""

    @abstractmethod
    async def fetch_candles(self, symbol: str = "XAUUSD", timeframe: str = "15m", limit: int = 100) -> CandleHistory:
        """Fetches normalized OHLCV candles."""
        pass

    @abstractmethod
    def get_provider_status(self) -> Dict[str, Any]:
        """Returns provider health, latency, freshness, and mode."""
        pass

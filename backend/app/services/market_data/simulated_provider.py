import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from app.services.market_data.base import MarketDataProvider
from app.schemas.candle import Candle, CandleHistory

class SimulatedMarketDataProvider(MarketDataProvider):
    """
    High-fidelity simulated Gold (XAUUSD) market data provider.
    Models realistic trending regimes, swing structures, Fair Value Gaps,
    and volatility spikes for local testing and deterministic demonstrations.
    """
    def __init__(self, seed: int = 42):
        self.provider_name = "Simulated Market Engine (XAUUSD)"
        self.last_update = datetime.now(timezone.utc)
        self.failure_count = 0
        self.seed = seed
        
    async def fetch_candles(self, symbol: str = "XAUUSD", timeframe: str = "15m", limit: int = 120) -> CandleHistory:
        np.random.seed(self.seed)
        now = datetime.now(timezone.utc)
        
        # Determine timeframe delta
        tf_minutes = 15
        if timeframe == "5m":
            tf_minutes = 5
        elif timeframe == "1h":
            tf_minutes = 60
        elif timeframe == "4h":
            tf_minutes = 240
            
        base_price = 2735.50
        candles: List[Candle] = []
        
        # Generate a realistic series with multiple swing cycles
        current_p = base_price
        t_start = now - timedelta(minutes=tf_minutes * limit)
        
        # Introduce a clean wave structure
        for i in range(limit):
            t_candle = t_start + timedelta(minutes=tf_minutes * i)
            
            # Trend component + random walk
            cycle = np.sin(i / 14.0) * 8.0 + (i * 0.12)
            noise = np.random.normal(0, 1.2)
            
            open_p = current_p
            close_p = open_p + (cycle * 0.05) + noise
            
            wick_high = abs(np.random.normal(0.8, 0.4))
            wick_low = abs(np.random.normal(0.8, 0.4))
            
            high_p = max(open_p, close_p) + wick_high
            low_p = min(open_p, close_p) - wick_low
            volume = float(np.random.randint(800, 3500))
            
            current_p = close_p
            
            candles.append(Candle(
                timestamp=t_candle,
                open=round(float(open_p), 2),
                high=round(float(high_p), 2),
                low=round(float(low_p), 2),
                close=round(float(close_p), 2),
                volume=volume
            ))
            
        self.last_update = now
        return CandleHistory(
            symbol=symbol.upper(),
            timeframe=timeframe,
            source="SIMULATED",
            last_updated=now,
            is_live=False,
            is_stale=False,
            candles=candles
        )

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "ONLINE",
            "feed_type": "SIMULATED",
            "is_live": False,
            "last_updated": self.last_update.isoformat(),
            "failure_count": self.failure_count,
            "latency_ms": 12
        }

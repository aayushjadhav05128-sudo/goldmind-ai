import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import io
from app.services.market_data.base import MarketDataProvider
from app.schemas.candle import Candle, CandleHistory

class CSVMarketDataProvider(MarketDataProvider):
    """
    Parses historical OHLCV data from CSV text or file buffer.
    Detects standard column mappings and normalizes timestamps to UTC.
    """
    def __init__(self, csv_content: Optional[str] = None):
        self.provider_name = "Uploaded CSV Historical Feed"
        self.last_update = datetime.now(timezone.utc)
        self.csv_content = csv_content
        self.cached_candles: List[Candle] = []
        if csv_content:
            self._parse_csv(csv_content)

    def _parse_csv(self, content: str):
        df = pd.read_csv(io.StringIO(content))
        # Normalize column names to lowercase
        col_map = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=col_map)
        
        # Identify timestamp column
        ts_col = None
        for candidate in ['timestamp', 'time', 'date', 'datetime']:
            if candidate in df.columns:
                ts_col = candidate
                break
                
        if ts_col is None:
            df['timestamp'] = [datetime.now(timezone.utc) for _ in range(len(df))]
        else:
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
            
        candles = []
        for _, row in df.iterrows():
            candles.append(Candle(
                timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0.0))
            ))
        # Ensure chronological order
        candles.sort(key=lambda c: c.timestamp)
        self.cached_candles = candles
        self.last_update = datetime.now(timezone.utc)

    async def fetch_candles(self, symbol: str = "XAUUSD", timeframe: str = "15m", limit: int = 100) -> CandleHistory:
        candles = self.cached_candles[-limit:] if self.cached_candles else []
        return CandleHistory(
            symbol=symbol.upper(),
            timeframe=timeframe,
            source="CSV_UPLOAD",
            last_updated=self.last_update,
            is_live=False,
            is_stale=False,
            candles=candles
        )

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "LOADED" if self.cached_candles else "EMPTY",
            "feed_type": "CSV_HISTORICAL",
            "is_live": False,
            "candle_count": len(self.cached_candles),
            "last_updated": self.last_update.isoformat()
        }

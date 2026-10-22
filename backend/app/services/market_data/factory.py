from typing import Dict, Optional
from app.services.market_data.base import MarketDataProvider
from app.services.market_data.live_provider import LiveMarketDataProvider
from app.services.market_data.simulated_provider import SimulatedMarketDataProvider
from app.services.market_data.csv_provider import CSVMarketDataProvider

class MarketDataFactory:
    """Manages active market data providers with automatic fallback and transparent labeling."""
    _instance: Optional['MarketDataFactory'] = None
    
    def __init__(self):
        self.live_provider = LiveMarketDataProvider()
        self.simulated_provider = SimulatedMarketDataProvider()
        self.csv_provider: Optional[CSVMarketDataProvider] = None
        self.use_live = True
        
    @classmethod
    def get_instance(cls) -> 'MarketDataFactory':
        if cls._instance is None:
            cls._instance = MarketDataFactory()
        return cls._instance

    def set_live_mode(self):
        self.use_live = True

    def set_simulated_mode(self):
        self.use_live = False

    def load_csv(self, content: str):
        self.csv_provider = CSVMarketDataProvider(content)

    def get_provider(self) -> MarketDataProvider:
        if self.csv_provider and self.csv_provider.cached_candles:
            return self.csv_provider
        if self.use_live:
            return self.live_provider
        return self.simulated_provider

market_data_factory = MarketDataFactory.get_instance()

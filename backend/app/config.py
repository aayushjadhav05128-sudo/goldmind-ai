from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "GoldMind AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./goldmind.db"
    
    # Webhook & Security
    TV_WEBHOOK_SECRET: str = os.getenv("TV_WEBHOOK_SECRET", "gm_secret_default_key")
    WEBHOOK_RATE_LIMIT_PER_MINUTE: int = 60
    
    # AI Engine Provider
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini")  # "gemini" | "openai" | "anthropic" | "deterministic"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    
    # Default Risk & Capital Protection Settings
    DEFAULT_ACCOUNT_CURRENCY: str = "INR"  # "INR" or "USD"
    DEFAULT_ACCOUNT_BALANCE: float = 2000.0  # Starting capital as low as ₹2,000
    DEFAULT_RISK_PER_TRADE_PCT: float = 0.5  # 0.5% default
    MAX_RISK_PER_TRADE_PCT: float = 1.0      # 1% hard cap
    MAX_DAILY_LOSS_PCT: float = 2.0          # 2% max daily drawdown limit
    MAX_CONCURRENT_TRADES: int = 1           # 1 concurrent planned trade default
    
    # Market & Forex Specifications (XAUUSD)
    USD_INR_RATE: float = 86.50              # Default INR per USD conversion rate
    XAUUSD_CONTRACT_SIZE: float = 100.0     # 100 troy oz per standard lot (configurable)
    BROKER_MIN_LOT: float = 0.01            # Minimum broker lot size (0.01 standard lot = 1 oz)
    BROKER_LOT_STEP: float = 0.01           # Minimum lot step increment
    DEFAULT_LEVERAGE: float = 100.0         # 1:100 leverage estimate for margin check
    ESTIMATED_SPREAD_PIPS: float = 2.5      # Estimated Gold spread in points ($0.25)
    ESTIMATED_COMMISSION_PER_LOT: float = 7.0 # $7 per round lot
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()

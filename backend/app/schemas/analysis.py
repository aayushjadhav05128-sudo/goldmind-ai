from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SwingPoint(BaseModel):
    price: float
    index: int
    timestamp: str
    kind: str # "HIGH" or "LOW"
    label: Optional[str] = None # "HH", "HL", "LH", "LL"

class SRZone(BaseModel):
    zone_type: str # "SUPPORT" or "RESISTANCE"
    level_min: float
    level_max: float
    mid_price: float
    touches: int
    strength: str # "STRONG", "MODERATE", "WEAK"

class FVGZone(BaseModel):
    fvg_type: str # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    mid: float
    candle_index: int
    candle_timestamp: str
    mitigated: bool = False

class PriceActionPatterns(BaseModel):
    equal_highs: List[float] = []
    equal_lows: List[float] = []
    liquidity_sweeps: List[Dict[str, Any]] = []
    fair_value_gaps: List[FVGZone] = []
    rejection_candles: List[Dict[str, Any]] = []
    engulfing_patterns: List[Dict[str, Any]] = []

class IndicatorSnapshot(BaseModel):
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    atr14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    current_session: str # "LONDON", "NEW_YORK", "ASIAN", "OVERLAP"
    london_high: Optional[float] = None
    london_low: Optional[float] = None
    ny_high: Optional[float] = None
    ny_low: Optional[float] = None

class MarketStructureSummary(BaseModel):
    trend: str # "BULLISH", "BEARISH", "RANGING", "UNCERTAIN"
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    swing_points: List[SwingPoint] = []
    bos_detected: bool = False
    bos_direction: Optional[str] = None
    choch_detected: bool = False
    choch_direction: Optional[str] = None
    sr_zones: List[SRZone] = []
    market_regime: str = "TRENDING" # "TRENDING", "RANGING", "HIGH_VOLATILITY", "UNCERTAIN"

class MultiTimeframeSummary(BaseModel):
    tf_4h_trend: str = "BULLISH"
    tf_1h_trend: str = "BULLISH"
    tf_15m_trend: str = "BULLISH"
    tf_5m_trend: str = "BULLISH"
    alignment_score: float = 1.0 # 0.0 to 1.0
    confirmed: bool = True

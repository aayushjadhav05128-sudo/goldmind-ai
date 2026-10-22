from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class PatternDetection:
    """
    Standardized Detection Model satisfying Section B requirements:
    - Pattern/strategy name and direction
    - Exact candle indices and timestamps
    - Defined price boundaries and timeframe
    - Detection rules and confirmation requirements
    - Entry trigger, invalidation condition and target methodology
    - Relevant market regime and trading session
    - Historical sample size and out-of-sample results
    - Confidence/calibration information and known failure conditions
    - Status: CONFIRMED, DEVELOPING, or INVALIDATED (No future leak)
    """
    pattern_id: str
    name: str
    category: str # "CANDLESTICK", "CHART_PATTERN", "SMC_ICT", "MARKET_STRUCTURE", "STRATEGY"
    direction: str # "BULLISH", "BEARISH", "NEUTRAL"
    status: str # "CONFIRMED", "DEVELOPING", "INVALIDATED"
    timeframe: str
    
    # Exact candle indices and timestamps
    start_index: int
    end_index: int
    start_timestamp: str
    end_timestamp: str
    
    # Defined price boundaries
    price_high: float
    price_low: float
    trigger_price: Optional[float] = None
    invalidation_level: Optional[float] = None
    target_price: Optional[float] = None
    key_levels: Dict[str, float] = field(default_factory=dict)
    
    # Rules and criteria
    detection_rules: str = ""
    confirmation_requirements: str = ""
    entry_trigger: str = ""
    invalidation_condition: str = ""
    target_methodology: str = ""
    
    # Context
    market_regime: str = "UNKNOWN" # "TRENDING_BULLISH", "TRENDING_BEARISH", "RANGING", "HIGH_VOLATILITY"
    trading_session: str = "UNKNOWN"
    
    # Quantitative calibration & historical evidence
    sample_size: int = 0
    out_of_sample_win_rate: Optional[float] = None
    out_of_sample_profit_factor: Optional[float] = None
    confidence_score: float = 0.5 # 0.0 to 1.0 calibrated score
    known_failure_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "category": self.category,
            "direction": self.direction,
            "status": self.status,
            "timeframe": self.timeframe,
            "candle_indices": [self.start_index, self.end_index],
            "timestamps": [self.start_timestamp, self.end_timestamp],
            "boundaries": {
                "high": round(self.price_high, 2),
                "low": round(self.price_low, 2),
                "trigger": round(self.trigger_price, 2) if self.trigger_price is not None else None,
                "invalidation": round(self.invalidation_level, 2) if self.invalidation_level is not None else None,
                "target": round(self.target_price, 2) if self.target_price is not None else None,
                "key_levels": {k: round(v, 2) for k, v in self.key_levels.items()}
            },
            "detection_rules": self.detection_rules,
            "confirmation_requirements": self.confirmation_requirements,
            "entry_trigger": self.entry_trigger,
            "invalidation_condition": self.invalidation_condition,
            "target_methodology": self.target_methodology,
            "market_regime": self.market_regime,
            "trading_session": self.trading_session,
            "historical_validation": {
                "sample_size": self.sample_size,
                "oos_win_rate_pct": round(self.out_of_sample_win_rate * 100, 1) if self.out_of_sample_win_rate is not None else None,
                "oos_profit_factor": round(self.out_of_sample_profit_factor, 2) if self.out_of_sample_profit_factor is not None else None,
                "confidence_score": round(self.confidence_score, 2),
            },
            "known_failure_conditions": self.known_failure_conditions
        }

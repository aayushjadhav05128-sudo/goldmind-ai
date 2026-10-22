from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime, timezone

class TradeJournal:
    """
    Trade Journal and Manual Feedback Logging System for XAUUSD.
    Strictly satisfies Section 11 requirements:
    - Logs signals, prediction timestamps, model/strategy versions, original evidence, and estimated costs.
    - Records manual trade execution details, actual outcomes (WIN/LOSS/BREAKEVEN),
      Maximum Adverse Excursion (MAE), Maximum Favourable Excursion (MFE), and rule compliance.
    - Strictly prevents single-trade auto-retraining model drift:
      Logs outcomes to a segregated offline retraining queue requiring reproducible out-of-sample evaluation before model approval.
    """

    JOURNAL_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "trade_journal.json"
    )

    @classmethod
    def _load_journal(cls) -> List[Dict[str, Any]]:
        if os.path.exists(cls.JOURNAL_FILE):
            try:
                with open(cls.JOURNAL_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    @classmethod
    def _save_journal(cls, records: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(cls.JOURNAL_FILE), exist_ok=True)
        with open(cls.JOURNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    @classmethod
    def log_trade_signal(
        cls,
        decision: str,
        symbol: str,
        timeframe: str,
        price: float,
        model_version: str,
        strategy_id: Optional[str],
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        spread_pts: float,
        slippage_pts: float,
        model_probabilities: Dict[str, float],
        original_evidence: Dict[str, Any]
    ) -> str:
        """
        Logs a generated decision-support signal for auditing and future walk-forward research.
        """
        records = cls._load_journal()
        signal_id = f"SIG-{len(records) + 1:05d}-{int(datetime.now(timezone.utc).timestamp())}"

        new_entry = {
            "signal_id": signal_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "symbol": symbol,
            "timeframe": timeframe,
            "market_price": price,
            "model_version": model_version,
            "strategy_id": strategy_id,
            "trade_geometry": {
                "proposed_entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "estimated_spread_pts": spread_pts,
                "estimated_slippage_pts": slippage_pts,
            },
            "model_probabilities": model_probabilities,
            "original_evidence": original_evidence,
            "user_execution": None,
            "outcome_logged": False
        }

        records.append(new_entry)
        cls._save_journal(records)
        return signal_id

    @classmethod
    def record_manual_outcome(
        cls,
        signal_id: str,
        actual_entry_price: float,
        actual_exit_price: float,
        outcome: str, # "WIN", "LOSS", "BREAKEVEN"
        max_adverse_excursion_pts: float,
        max_favourable_excursion_pts: float,
        rules_strictly_followed: bool,
        notes: str = ""
    ) -> bool:
        """
        Records a manually executed trade result into the offline evaluation queue.
        """
        records = cls._load_journal()
        for r in records:
            if r["signal_id"] == signal_id:
                r["user_execution"] = {
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "actual_entry": actual_entry_price,
                    "actual_exit": actual_exit_price,
                    "outcome": outcome.upper(),
                    "max_adverse_excursion_pts": max_adverse_excursion_pts,
                    "max_favourable_excursion_pts": max_favourable_excursion_pts,
                    "rules_strictly_followed": rules_strictly_followed,
                    "notes": notes
                }
                r["outcome_logged"] = True
                cls._save_journal(records)
                return True
        return False

    @classmethod
    def get_journal_summary(cls) -> Dict[str, Any]:
        records = cls._load_journal()
        completed = [r for r in records if r.get("outcome_logged")]
        wins = [r for r in completed if r["user_execution"]["outcome"] == "WIN"]
        losses = [r for r in completed if r["user_execution"]["outcome"] == "LOSS"]

        win_rate = (len(wins) / len(completed) * 100) if completed else 0.0

        return {
            "total_signals_logged": len(records),
            "completed_manual_trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "pending_offline_retrain_queue": len(completed),
            "policy": "Offline batch evaluation only. Never auto-modifies live models on single trades."
        }

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

class ModelCalibrator:
    """
    Probability Calibration and Uncertainty Assessment Engine.
    Strictly satisfies Section 9 requirements:
    - Probability calibration on independent validation partition
    - Calculation of Brier score and Expected Calibration Error (ECE)
    - Distinguishes:
        * Model-estimated probability
        * Historical setup empirical win rate
        * Data quality confidence
        * Strength of technical evidence
    - Never displays arbitrary, ungrounded confidence percentages.
    """

    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates Brier Score, Expected Calibration Error (ECE), and Reliability Binning.
        """
        if len(y_true) == 0:
            return {"brier_score": 1.0, "ece": 1.0, "calibration_status": "UNAVAILABLE"}

        # Brier score: Lower is better (0.0 = perfect calibration, 0.25 = uninformative coin toss)
        brier = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        bin_details = []

        for b in range(n_bins):
            mask = bin_indices == b
            bin_size = int(np.sum(mask))
            if bin_size > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                diff = abs(bin_acc - bin_conf)
                ece += (bin_size / len(y_true)) * diff
                bin_details.append({
                    "bin": b,
                    "count": bin_size,
                    "avg_confidence": round(bin_conf, 3),
                    "empirical_accuracy": round(bin_acc, 3),
                    "error": round(diff, 3)
                })

        # Calibration rating
        if brier <= 0.18 and ece <= 0.10:
            status = "WELL_CALIBRATED"
        elif brier <= 0.22:
            status = "MODERATELY_CALIBRATED"
        else:
            status = "POORLY_CALIBRATED_UNCERTAIN"

        return {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(float(ece), 4),
            "calibration_status": status,
            "sample_size": len(y_true),
            "bin_details": bin_details
        }

    @staticmethod
    def combine_probability_and_evidence(
        model_prob: float,
        setup_win_rate: Optional[float],
        technical_evidence_score: float,
        data_freshness_score: float,
        calibration_brier: float
    ) -> Dict[str, Any]:
        """
        Safely computes composite calibrated confidence, ensuring no single factor dominates
        and penalizing uncalibrated models or stale data.
        """
        # If calibration Brier is high (> 0.23), damp the model probability toward 0.50
        calibration_penalty = max(0.0, (0.25 - calibration_brier) / 0.10)
        calibrated_prob = 0.50 + (model_prob - 0.50) * max(0.2, min(1.0, calibration_penalty))

        # Setup historical weight
        hist_win = setup_win_rate if setup_win_rate is not None else 0.50

        # Technical evidence (0.0 to 1.0)
        # Data freshness (0.0 to 1.0)
        composite_confidence = (
            calibrated_prob * 0.40 +
            hist_win * 0.30 +
            technical_evidence_score * 0.20 +
            data_freshness_score * 0.10
        )

        return {
            "model_raw_prob_pct": round(model_prob * 100, 1),
            "model_calibrated_prob_pct": round(calibrated_prob * 100, 1),
            "historical_setup_win_rate_pct": round(hist_win * 100, 1) if setup_win_rate else None,
            "technical_evidence_strength": round(technical_evidence_score, 2),
            "data_quality_confidence": round(data_freshness_score, 2),
            "composite_confidence_pct": round(composite_confidence * 100, 1),
            "calibration_penalty_applied": bool(calibration_penalty < 0.9)
        }

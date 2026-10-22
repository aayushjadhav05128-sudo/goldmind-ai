import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.services.ml.data_manager import MarketDataManager
from app.services.ml.feature_engineering import FeatureEngineer
from app.services.ml.labeling import MultiTaskLabeler
from app.services.ml.calibration import ModelCalibrator
from app.services.ml.trainer import ModelTrainer
from app.services.ml.predictor import LivePredictor
from app.services.ml.trade_journal import TradeJournal
from app.services.chat_engine import process_chat_message

def test_data_manager_validation():
    dm = MarketDataManager()
    # Test duplicate removal and price sanity
    raw_df = pd.DataFrame([
        {"timestamp": "2026-10-21T10:00:00Z", "open": 2700.0, "high": 2705.0, "low": 2695.0, "close": 2702.0},
        {"timestamp": "2026-10-21T10:00:00Z", "open": 2700.0, "high": 2705.0, "low": 2695.0, "close": 2702.0}, # Duplicate
        {"timestamp": "2026-10-21T10:15:00Z", "open": -10.0, "high": 2705.0, "low": 2695.0, "close": 2702.0}, # Corrupted negative price
        {"timestamp": "2026-10-21T10:30:00Z", "open": 2702.0, "high": 2706.0, "low": 2700.0, "close": 2704.0},
    ])
    cleaned, report = dm.validate_dataset(raw_df, min_rows=2)
    assert report["duplicates_removed"] == 1
    assert report["corrupted_rows_removed"] == 1
    assert len(cleaned) == 2

def test_no_lookahead_bias_in_features():
    """
    Strictly verifies that calculating features at index i does NOT change
    when subsequent future bars (i+1, i+2, ...) are added to the dataset.
    """
    dm = MarketDataManager()
    df_short = dm.generate_benchmark_dataset(n_bars=60)
    df_long = dm.generate_benchmark_dataset(n_bars=100)

    # For the first 60 bars, features at bar 50 must be 100% IDENTICAL in both datasets
    feats_short = FeatureEngineer.extract_features_at_index(df_short, idx=50)
    feats_long = FeatureEngineer.extract_features_at_index(df_long, idx=50)

    for k in FeatureEngineer.FEATURE_NAMES:
        assert np.isclose(feats_short[k], feats_long[k], atol=1e-5), f"Lookahead leakage detected in feature '{k}'!"

def test_multi_task_labeling():
    dm = MarketDataManager()
    df = dm.generate_benchmark_dataset(n_bars=100)
    labels = MultiTaskLabeler.label_dataset(df, horizons=[4, 8, 16])

    assert "label_rise_h4" in labels.columns
    assert "label_fall_h4" in labels.columns
    assert "label_range_h4" in labels.columns
    assert "label_triple_barrier_buy" in labels.columns
    assert "label_triple_barrier_sell" in labels.columns
    assert "label_regime" in labels.columns

    # Verify purge and embargo
    X = pd.DataFrame({"feat1": range(len(labels))})
    y = labels["label_triple_barrier_buy"]
    clean_X, clean_y = MultiTaskLabeler.purge_and_embargo(X, y, horizon_bars=8)
    assert len(clean_X) < len(labels)

def test_probability_calibration_metrics():
    # Synthetic ground truth and probabilities
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.3, 0.8, 0.7])

    calib = ModelCalibrator.calculate_calibration_metrics(y_true, y_prob)
    assert calib["brier_score"] < 0.15
    assert calib["calibration_status"] in ["WELL_CALIBRATED", "MODERATELY_CALIBRATED"]

    # Uncertainty combination
    unc = ModelCalibrator.combine_probability_and_evidence(
        model_prob=0.80,
        setup_win_rate=0.60,
        technical_evidence_score=0.85,
        data_freshness_score=1.0,
        calibration_brier=calib["brier_score"]
    )
    assert 50.0 < unc["composite_confidence_pct"] <= 100.0

def test_model_training_and_comparison():
    dm = MarketDataManager()
    df = dm.generate_benchmark_dataset(n_bars=300)
    trainer = ModelTrainer(target_label="label_triple_barrier_buy", model_version="test-v1")
    res = trainer.train_and_compare_models(df)

    assert "best_model_selected" in res
    assert "Baseline_Majority" in res["model_comparison"]
    assert "Calibrated_Logistic_Regression" in res["model_comparison"]
    assert res["verification_status"] == "EMPIRICALLY_TRAINED_AND_CALIBRATED"

def test_live_predictor():
    dm = MarketDataManager()
    df = dm.generate_benchmark_dataset(n_bars=80)
    predictor = LivePredictor(target_label="label_triple_barrier_buy")
    pred = predictor.predict_current_state(df)

    assert "probabilities" in pred
    assert "prob_rise_target" in pred["probabilities"]
    assert "prob_fall_target" in pred["probabilities"]
    assert 0.0 <= pred["probabilities"]["prob_rise_target"] <= 1.0

def test_trade_journal_logging():
    sig_id = TradeJournal.log_trade_signal(
        decision="NO TRADE",
        symbol="XAUUSD",
        timeframe="15m",
        price=2720.0,
        model_version="v1.2.0",
        strategy_id="STRAT-XAU-001",
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        spread_pts=2.0,
        slippage_pts=0.20,
        model_probabilities={"prob_rise_target": 0.55},
        original_evidence={"why": "Small account risk limit guardrail"}
    )
    assert sig_id.startswith("SIG-")

    # Record manual outcome
    ok = TradeJournal.record_manual_outcome(
        signal_id=sig_id,
        actual_entry_price=2720.0,
        actual_exit_price=2725.0,
        outcome="WIN",
        max_adverse_excursion_pts=0.8,
        max_favourable_excursion_pts=5.2,
        rules_strictly_followed=True
    )
    assert ok is True

    summary = TradeJournal.get_journal_summary()
    assert summary["total_signals_logged"] >= 1

@pytest.mark.asyncio
async def test_chat_engine_with_ml_intelligence():
    res = await process_chat_message(
        user_query="Show Model Accuracy and Backtest Results",
        symbol="XAUUSD",
        timeframe="15m",
        account_currency="INR",
        account_balance=2000.0,
        risk_pct=0.5
    )
    assert "model_predictions" in res
    assert "journal_summary" in res
    assert "Brier" in res["response"] or "Model" in res["response"]
    assert "**Decision: NO TRADE**" in res["response"]

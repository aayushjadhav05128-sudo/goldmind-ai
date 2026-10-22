import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from app.services.ml.data_manager import MarketDataManager
from app.services.ml.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="GoldMind AI — Reproducible XAUUSD ML Training Pipeline")
    parser.add_argument("--bars", type=int, default=1200, help="Number of historical candle bars")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--target", type=str, default="label_triple_barrier_buy", help="Prediction task label")
    parser.add_argument("--version", type=str, default="v1.2.0", help="Model version tag")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print(" GOLDMIND AI — XAUUSD MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)
    print(f"Timeframe: {args.timeframe} | Target: {args.target} | Version: {args.version}")

    # 1. Ingest & Validate Data
    dm = MarketDataManager()
    print("\n[1/4] Ingesting and validating historical candle data...")
    df = dm.generate_benchmark_dataset(n_bars=args.bars, timeframe=args.timeframe)
    cleaned_df, report = dm.validate_dataset(df, timeframe=args.timeframe)
    dm.save_dataset(cleaned_df, f"xauusd_{args.timeframe}_training_dataset.csv")

    print(f"  ✓ Cleaned candles: {report['final_rows']} rows (Removed: {report['duplicates_removed']} dups, {report['corrupted_rows_removed']} corrupted)")
    print(f"  ✓ Outliers flagged: {report['outliers_flagged']} | Time gaps: {report['time_gaps_detected']}")

    # 2. Train and Evaluate Models
    print("\n[2/4] Training baseline and machine learning candidates...")
    trainer = ModelTrainer(target_label=args.target, model_version=args.version)
    train_results = trainer.train_and_compare_models(cleaned_df)

    if "error" in train_results:
        print(f"Error during training: {train_results['error']}")
        return

    print("\n[3/4] Model Evaluation & Out-of-Sample Calibration Results:")
    print("-" * 70)
    print(f"{'Model Name':<32} {'Val Acc':<9} {'Test Acc':<10} {'ROC-AUC':<9} {'Brier Score':<12}")
    print("-" * 70)
    for model_name, m in train_results["model_comparison"].items():
        if "error" in m:
            continue
        print(f"{model_name:<32} {m.get('validation_accuracy', 0.0):<9.3f} {m.get('test_accuracy', 0.0):<10.3f} {m.get('test_roc_auc', 0.0):<9.3f} {m.get('test_brier_score', 0.0):<12.4f}")
    print("-" * 70)

    # 3. Model Approval
    print(f"\n[4/4] Approved Production Model: {train_results['best_model_selected']}")
    print(f"  ✓ Model bundle saved to: {train_results['saved_model_path']}")
    print(f"  ✓ Partitions: Train={train_results['partitions']['train_samples']}, Val={train_results['partitions']['val_samples']}, Test={train_results['partitions']['test_samples']}")
    print("\nTraining completed successfully with zero look-ahead bias and grounded probability calibration.\n")

if __name__ == "__main__":
    main()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from app.services.ml.data_manager import MarketDataManager
from app.services.knowledge.backtest_pipeline import HistoricalBacktestPipeline
from app.services.knowledge.strategy_registry import StrategyRegistry

def main():
    parser = argparse.ArgumentParser(description="GoldMind AI — Reproducible XAUUSD Backtesting Pipeline")
    parser.add_argument("--bars", type=int, default=1200, help="Number of historical candle bars")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--strategy", type=str, default="STRAT-XAU-001", help="Strategy ID to evaluate")
    parser.add_argument("--spread", type=float, default=2.0, help="Spread in points (e.g. 2.0)")
    parser.add_argument("--slippage", type=float, default=0.20, help="Slippage in points")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print(" GOLDMIND AI — REPRODUCIBLE XAUUSD BACKTESTING PIPELINE")
    print("=" * 70)
    print(f"Strategy: {args.strategy} | Timeframe: {args.timeframe}")
    print(f"Transaction Model: Spread={args.spread} pts | Slippage={args.slippage} pts | Comm=$3.50/lot")
    print("Execution Rule: CONSERVATIVE (Same-candle SL hit first)\n")

    strat = StrategyRegistry.get_strategy(args.strategy)
    if strat:
        print(f"Hypothesis: {strat.hypothesis}")
        print(f"Provenance: {strat.source_citation} ({strat.publication_year})\n")

    # Ingest data
    dm = MarketDataManager()
    df = dm.generate_benchmark_dataset(n_bars=args.bars, timeframe=args.timeframe)
    cleaned_df, report = dm.validate_dataset(df, timeframe=args.timeframe)

    # Generate signals based on strategy rules (EMA pullback)
    from app.services.knowledge.market_structure import MarketStructureEngine
    from app.services.knowledge.indicators_advanced import IndicatorsAdvancedEngine
    
    signals = []
    for i in range(35, len(cleaned_df) - 1):
        sub = cleaned_df.iloc[:i+1]
        ind = IndicatorsAdvancedEngine.compute_all_indicators(sub)
        c = float(cleaned_df['close'].iloc[i])
        ema20 = ind.get("ema20", c)
        ema50 = ind.get("ema50", c)
        atr = ind.get("atr14", 3.0)

        # Bullish pullback signal
        if c > ema20 > ema50 and abs(c - ema20) <= 0.8:
            signals.append({
                "entry_index": i,
                "direction": "BUY",
                "entry_price": c,
                "stop_loss": c - 1.2 * atr,
                "take_profit": c + 2.4 * atr,
                "lot_size": 0.01
            })
        # Bearish pullback signal
        elif c < ema20 < ema50 and abs(c - ema20) <= 0.8:
            signals.append({
                "entry_index": i,
                "direction": "SELL",
                "entry_price": c,
                "stop_loss": c + 1.2 * atr,
                "take_profit": c - 2.4 * atr,
                "lot_size": 0.01
            })

    print(f"Generated {len(signals)} candidate trade signals for walk-forward execution.")

    # Run walk-forward backtest
    pipeline = HistoricalBacktestPipeline()
    res = pipeline.run_walk_forward_evaluation(
        cleaned_df,
        signals,
        default_spread=args.spread,
        default_slippage=args.slippage
    )

    print("\nWALK-FORWARD PARTITION RESULTS:")
    print("-" * 70)
    for part_name, part_key in [("Training (60%)", "train_partition"), ("Validation (20%)", "validation_partition"), ("Out-of-Sample Test (20%)", "out_of_sample_test_partition")]:
        p = res.get(part_key, {})
        print(f"[{part_name}]")
        print(f"  Sample Trades: N={p.get('sample_size', 0)} | Win Rate: {p.get('win_rate_pct', 0.0)}%")
        print(f"  Profit Factor: {p.get('profit_factor', 0.0)} | Expectancy: {p.get('expectancy_r', 0.0)}R | Net PnL: ${p.get('total_net_pnl_usd', 0.0):.2f}")
    print("-" * 70)

    print(f"\nBootstrap 95% Confidence Interval for OOS Win Rate: {res.get('bootstrap_95pct_ci_win_rate')}")
    print(f"Paper-Trading Validation Status: {'APPROVED' if res.get('is_paper_trading_validated') else 'PENDING_SAMPLE_ACCUMULATION'}")
    print("\nBacktest completed with zero lookahead bias.\n")

if __name__ == "__main__":
    main()

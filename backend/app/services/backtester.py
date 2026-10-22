import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.services.technical.indicators import compute_all_indicators
from app.services.technical.market_structure import analyze_market_structure
from app.services.technical.price_action import analyze_price_action

class BacktestTrade:
    def __init__(
        self,
        entry_idx: int,
        entry_time: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        risk_dist: float
    ):
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.risk_dist = risk_dist
        
        self.exit_idx: Optional[int] = None
        self.exit_time: Optional[str] = None
        self.exit_price: Optional[float] = None
        self.outcome: Optional[str] = None # "WIN", "LOSS", "BREAKEVEN"
        self.r_multiple: float = 0.0
        self.net_pnl: float = 0.0

def run_backtest(
    df: pd.DataFrame,
    spread_points: float = 0.25, # $0.25 on gold
    slippage_points: float = 0.10,
    contract_size: float = 100.0,
    lot_size: float = 0.01,
    split_ratios: tuple = (0.6, 0.2, 0.2)
) -> Dict[str, Any]:
    """
    Backtesting engine enforcing chronological progression with ZERO look-ahead bias.
    Executes conservative same-candle resolution (SL hit first if both SL and TP breached).
    Calculates out-of-sample metrics for Train, Validation, and Test partitions.
    """
    n = len(df)
    if n < 60:
        return {"error": "Insufficient candles for backtesting. Minimum 60 required."}

    # Pre-calculate indicator signals up to current bar to prevent lookahead
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))

    all_trades: List[BacktestTrade] = []
    active_trade: Optional[BacktestTrade] = None

    # Step bar-by-bar
    for i in range(35, n):
        curr_bar = df.iloc[i]
        bar_open = float(curr_bar['open'])
        bar_high = float(curr_bar['high'])
        bar_low = float(curr_bar['low'])
        bar_close = float(curr_bar['close'])
        bar_time = str(curr_bar['timestamp']) if 'timestamp' in curr_bar else str(i)

        # 1. Manage Active Trade (Conservative Resolution)
        if active_trade is not None:
            hit_tp = False
            hit_sl = False

            if active_trade.direction == "BUY":
                if bar_low <= active_trade.stop_loss:
                    hit_sl = True
                if bar_high >= active_trade.take_profit_1:
                    hit_tp = True

                # Conservative Rule: If both hit within the same candle, assume SL hit first
                if hit_sl and hit_tp:
                    hit_sl = True
                    hit_tp = False

                if hit_sl:
                    active_trade.exit_idx = i
                    active_trade.exit_time = bar_time
                    active_trade.exit_price = active_trade.stop_loss - slippage_points
                    active_trade.outcome = "LOSS"
                    active_trade.r_multiple = -1.0
                    loss_dist = abs(active_trade.entry_price - active_trade.exit_price)
                    active_trade.net_pnl = -(loss_dist * contract_size * lot_size)
                    all_trades.append(active_trade)
                    active_trade = None
                elif hit_tp:
                    active_trade.exit_idx = i
                    active_trade.exit_time = bar_time
                    active_trade.exit_price = active_trade.take_profit_1 - slippage_points
                    active_trade.outcome = "WIN"
                    gain_dist = abs(active_trade.exit_price - active_trade.entry_price)
                    active_trade.r_multiple = round(gain_dist / active_trade.risk_dist, 2) if active_trade.risk_dist > 0 else 1.5
                    active_trade.net_pnl = gain_dist * contract_size * lot_size
                    all_trades.append(active_trade)
                    active_trade = None

            elif active_trade.direction == "SELL":
                if bar_high >= active_trade.stop_loss:
                    hit_sl = True
                if bar_low <= active_trade.take_profit_1:
                    hit_tp = True

                # Conservative Rule: Assume SL hit first
                if hit_sl and hit_tp:
                    hit_sl = True
                    hit_tp = False

                if hit_sl:
                    active_trade.exit_idx = i
                    active_trade.exit_time = bar_time
                    active_trade.exit_price = active_trade.stop_loss + slippage_points
                    active_trade.outcome = "LOSS"
                    active_trade.r_multiple = -1.0
                    loss_dist = abs(active_trade.exit_price - active_trade.entry_price)
                    active_trade.net_pnl = -(loss_dist * contract_size * lot_size)
                    all_trades.append(active_trade)
                    active_trade = None
                elif hit_tp:
                    active_trade.exit_idx = i
                    active_trade.exit_time = bar_time
                    active_trade.exit_price = active_trade.take_profit_1 + slippage_points
                    active_trade.outcome = "WIN"
                    gain_dist = abs(active_trade.entry_price - active_trade.exit_price)
                    active_trade.r_multiple = round(gain_dist / active_trade.risk_dist, 2) if active_trade.risk_dist > 0 else 1.5
                    active_trade.net_pnl = gain_dist * contract_size * lot_size
                    all_trades.append(active_trade)
                    active_trade = None

        # 2. Check for New Signal if no active trade
        if active_trade is None and i < n - 1:
            # Look only at history up to previous bar [0 : i]
            sub_df = df.iloc[max(0, i - 40): i].copy()
            df_ind = compute_all_indicators(sub_df)
            
            c_prev = float(df_ind['close'].iloc[-1])
            ema20 = float(df_ind['ema20'].iloc[-1])
            ema50 = float(df_ind['ema50'].iloc[-1])
            rsi = float(df_ind['rsi14'].iloc[-1])
            atr = float(df_ind['atr14'].iloc[-1]) if not np.isnan(df_ind['atr14'].iloc[-1]) else 3.5

            # Entry Trigger on bar open
            if c_prev > ema20 > ema50 and 50 <= rsi <= 65:
                # Buy signal
                entry_exec = bar_open + spread_points + slippage_points
                sl_exec = entry_exec - (atr * 1.5)
                tp1_exec = entry_exec + (atr * 3.0) # 1:2 RR
                risk_d = entry_exec - sl_exec
                
                active_trade = BacktestTrade(
                    entry_idx=i,
                    entry_time=bar_time,
                    direction="BUY",
                    entry_price=entry_exec,
                    stop_loss=sl_exec,
                    take_profit_1=tp1_exec,
                    take_profit_2=entry_exec + (atr * 4.5),
                    risk_dist=risk_d
                )
            elif c_prev < ema20 < ema50 and 35 <= rsi <= 50:
                # Sell signal
                entry_exec = bar_open - spread_points - slippage_points
                sl_exec = entry_exec + (atr * 1.5)
                tp1_exec = entry_exec - (atr * 3.0) # 1:2 RR
                risk_d = sl_exec - entry_exec
                
                active_trade = BacktestTrade(
                    entry_idx=i,
                    entry_time=bar_time,
                    direction="SELL",
                    entry_price=entry_exec,
                    stop_loss=sl_exec,
                    take_profit_1=tp1_exec,
                    take_profit_2=entry_exec - (atr * 4.5),
                    risk_dist=risk_d
                )

    # Compute Metrics across Splits
    def compute_partition_metrics(trades: List[BacktestTrade]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "expectancy_r": 0.0, "total_net_pnl": 0.0,
                "max_drawdown_pct": 0.0
            }
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        total_pnl = sum([t.net_pnl for t in trades])
        gross_profit = sum([t.net_pnl for t in wins]) if wins else 0.0
        gross_loss = abs(sum([t.net_pnl for t in losses])) if losses else 0.0
        
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = round((len(wins) / len(trades)) * 100.0, 1)
        r_multiples = [t.r_multiple for t in trades]
        expectancy = round(float(np.mean(r_multiples)), 2) if r_multiples else 0.0
        
        # Drawdown calculation
        equity_curve = np.cumsum([t.net_pnl for t in trades])
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = peak - equity_curve
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy,
            "total_net_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 2)
        }

    train_trades = [t for t in all_trades if t.entry_idx < train_end]
    val_trades = [t for t in all_trades if train_end <= t.entry_idx < val_end]
    test_trades = [t for t in all_trades if t.entry_idx >= val_end]

    return {
        "overall": compute_partition_metrics(all_trades),
        "in_sample_train": compute_partition_metrics(train_trades),
        "validation": compute_partition_metrics(val_trades),
        "out_of_sample_test": compute_partition_metrics(test_trades),
        "recent_trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "outcome": t.outcome,
                "r_multiple": t.r_multiple,
                "net_pnl": round(t.net_pnl, 2)
            }
            for t in all_trades[-15:]
        ]
    }

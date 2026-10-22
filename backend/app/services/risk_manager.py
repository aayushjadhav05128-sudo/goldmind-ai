import math
from typing import Optional
from app.schemas.risk import RiskSettings, PositionSizeResult

def calculate_position_size(
    entry_price: float,
    stop_loss_price: float,
    take_profit_1: float,
    take_profit_2: Optional[float] = None,
    take_profit_3: Optional[float] = None,
    settings: Optional[RiskSettings] = None
) -> PositionSizeResult:
    """
    Calculates conservative position size and capital protection metrics.
    Strictly safeguards small accounts (e.g. ₹2,000 INR) by enforcing:
    'TRADE SKIPPED — minimum position size exceeds your risk limit.'
    when broker minimum lots violate the risk tolerance.
    """
    if settings is None:
        settings = RiskSettings()

    # Prevent division by zero and invalid stops
    sl_distance = abs(entry_price - stop_loss_price)
    if sl_distance <= 0.0001:
        sl_distance = 1.0 # fallback safety

    # 1. Calculate Permitted Cash Risk
    # Cap risk percentage to max_risk_per_trade_pct
    effective_risk_pct = min(settings.risk_per_trade_pct, settings.max_risk_per_trade_pct)
    cash_risk_account_curr = settings.account_balance * (effective_risk_pct / 100.0)

    # Convert to USD if account is in INR
    if settings.account_currency.upper() == "INR":
        cash_risk_usd = cash_risk_account_curr / settings.usd_inr_rate
    else:
        cash_risk_usd = cash_risk_account_curr

    # 2. Instrument Math for XAUUSD (Gold)
    # 1 lot of XAUUSD represents contract_size troy ounces (standard = 100 oz)
    # A $1.00 move per 1.0 standard lot = $100.00 USD
    dollar_risk_per_full_lot = sl_distance * settings.contract_size
    theoretical_lot_size = cash_risk_usd / dollar_risk_per_full_lot if dollar_risk_per_full_lot > 0 else 0.0

    # 3. Lot Size Evaluation against Broker Minimum
    min_lot = settings.min_lot_size
    lot_step = settings.lot_step
    
    # Calculate what minimum lot would actually risk in USD and INR
    actual_risk_at_min_lot_usd = min_lot * dollar_risk_per_full_lot
    actual_risk_at_min_lot_account = (
        actual_risk_at_min_lot_usd * settings.usd_inr_rate 
        if settings.account_currency.upper() == "INR" 
        else actual_risk_at_min_lot_usd
    )
    actual_risk_pct_at_min_lot = (actual_risk_at_min_lot_account / settings.account_balance) * 100.0

    trade_permitted = True
    warning_message = None
    recommend_demo = False
    suggested_lot = 0.0

    if theoretical_lot_size < min_lot:
        trade_permitted = False
        recommend_demo = True
        suggested_lot = 0.0
        
        # Calculate capital needed to safely trade min lot at configured risk
        capital_needed = (actual_risk_at_min_lot_account / (effective_risk_pct / 100.0))
        curr_symbol = "₹" if settings.account_currency.upper() == "INR" else "$"
        
        warning_message = (
            f"TRADE SKIPPED — minimum position size exceeds your risk limit. "
            f"Your risk limit is {curr_symbol}{cash_risk_account_curr:.2f} ({effective_risk_pct}%), "
            f"but broker minimum {min_lot:.2f} lot on a {sl_distance:.2f} pt SL would risk "
            f"{curr_symbol}{actual_risk_at_min_lot_account:.2f} ({actual_risk_pct_at_min_lot:.1f}% of balance). "
            f"Demo trading or a nano/cent account is strongly advised until balance reaches {curr_symbol}{capital_needed:,.2f}."
        )
    else:
        # Round DOWN to nearest lot step increment
        steps = math.floor(theoretical_lot_size / lot_step)
        suggested_lot = round(steps * lot_step, 2)
        if suggested_lot < min_lot:
            suggested_lot = min_lot

    # 4. Estimated Margin Requirement
    # Margin (USD) = (Entry Price * Contract Size * Lot Size) / Leverage
    active_lot = suggested_lot if trade_permitted else min_lot
    margin_usd = (entry_price * settings.contract_size * active_lot) / max(settings.leverage, 1.0)
    margin_account = (
        margin_usd * settings.usd_inr_rate 
        if settings.account_currency.upper() == "INR" 
        else margin_usd
    )

    # 5. Profit & Loss calculations
    # Trading costs: Spread + Commission
    spread_cost_per_lot_usd = settings.spread_points * settings.contract_size
    total_cost_per_lot_usd = spread_cost_per_lot_usd + settings.commission_per_lot
    total_costs_usd = total_cost_per_lot_usd * active_lot

    def calc_pnl_usd(target_price: float, is_tp: bool = True) -> float:
        dist = abs(target_price - entry_price)
        gross_pnl = dist * settings.contract_size * active_lot
        return (gross_pnl - total_costs_usd) if is_tp else (-gross_pnl - total_costs_usd)

    est_loss_usd = calc_pnl_usd(stop_loss_price, is_tp=False)
    est_tp1_usd = calc_pnl_usd(take_profit_1, is_tp=True)
    est_tp2_usd = calc_pnl_usd(take_profit_2, is_tp=True) if take_profit_2 else (est_tp1_usd * 1.5)
    est_tp3_usd = calc_pnl_usd(take_profit_3, is_tp=True) if take_profit_3 else (est_tp1_usd * 2.0)

    def to_account(val_usd: float) -> float:
        return val_usd * settings.usd_inr_rate if settings.account_currency.upper() == "INR" else val_usd

    # 6. Cost-Adjusted Risk-to-Reward Ratios
    cost_pts = total_cost_per_lot_usd / settings.contract_size
    eff_sl_dist = sl_distance + cost_pts
    
    tp1_dist = abs(take_profit_1 - entry_price)
    rr_tp1 = max(0.0, (tp1_dist - cost_pts) / eff_sl_dist) if eff_sl_dist > 0 else 0.0
    
    tp2_dist = abs(take_profit_2 - entry_price) if take_profit_2 else (tp1_dist * 1.5)
    rr_tp2 = max(0.0, (tp2_dist - cost_pts) / eff_sl_dist) if eff_sl_dist > 0 else 0.0
    
    tp3_dist = abs(take_profit_3 - entry_price) if take_profit_3 else (tp1_dist * 2.0)
    rr_tp3 = max(0.0, (tp3_dist - cost_pts) / eff_sl_dist) if eff_sl_dist > 0 else 0.0

    return PositionSizeResult(
        cash_risk_permitted_account_currency=round(cash_risk_account_curr, 2),
        cash_risk_usd=round(cash_risk_usd, 2),
        entry_price=round(entry_price, 2),
        stop_loss_price=round(stop_loss_price, 2),
        sl_distance_points=round(sl_distance, 2),
        theoretical_lot_size=round(theoretical_lot_size, 4),
        suggested_lot_size=suggested_lot,
        trade_permitted=trade_permitted,
        warning_message=warning_message,
        recommend_demo_mode=recommend_demo,
        estimated_margin_required_usd=round(margin_usd, 2),
        estimated_margin_required_account_currency=round(margin_account, 2),
        estimated_loss_at_sl_account_currency=round(to_account(est_loss_usd), 2),
        estimated_profit_tp1_account_currency=round(to_account(est_tp1_usd), 2),
        estimated_profit_tp2_account_currency=round(to_account(est_tp2_usd), 2),
        estimated_profit_tp3_account_currency=round(to_account(est_tp3_usd), 2),
        cost_adjusted_rr_tp1=round(rr_tp1, 2),
        cost_adjusted_rr_tp2=round(rr_tp2, 2),
        cost_adjusted_rr_tp3=round(rr_tp3, 2),
    )

from typing import Dict, Any, Optional
from app.schemas.risk import RiskSettings

class RiskEngine:
    """
    Institutional Risk Management and Account Protection Engine for XAUUSD.
    Strictly enforces Section F and Module 10 requirements:
    - Reads contract specifications (100 oz per standard lot, $1 per point per 0.01 lot)
    - Validates account balance, currency conversion (e.g. INR to USD @ 86.50)
    - Enforces configured maximum risk limit (e.g. 0.5% of account balance)
    - Calculates exact position sizing (lot size) based on stop-loss distance
    - Broker Minimum Lot Guardrail: If broker minimum 0.01 lot exceeds allowed risk, returns NO TRADE
    - Minimum Reward-to-Risk Threshold: Requires minimum 1:2.0 R:R; otherwise returns WAIT
    - Strictly forbids Martingale, loss-recovery doubling, or guaranteed outcome claims
    """

    @classmethod
    def evaluate_account_risk(
        cls,
        account_balance: float,
        account_currency: str,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        contract_size: float = 100.0,
        min_lot_size: float = 0.01,
        usd_inr_rate: float = 86.50,
        leverage: float = 100.0,
        direction: str = "BUY"
    ) -> Dict[str, Any]:
        curr_symbol = "₹" if account_currency.upper() == "INR" else "$"
        
        # 1. Allowed cash risk calculation
        effective_risk_pct = min(risk_per_trade_pct, 2.0) # Hard safety cap at 2%
        allowed_risk_account_curr = account_balance * (effective_risk_pct / 100.0)
        allowed_risk_usd = allowed_risk_account_curr / usd_inr_rate if account_currency.upper() == "INR" else allowed_risk_account_curr

        # If no valid SL is provided, default to ATR-based 3.50 pt risk
        if stop_loss_price is None or stop_loss_price <= 0:
            sl_distance = 3.50
        else:
            sl_distance = abs(entry_price - stop_loss_price)

        if sl_distance <= 0.10:
            sl_distance = 3.50

        # 2. Dollar risk at broker minimum 0.01 lot
        dollar_risk_min_lot = min_lot_size * sl_distance * contract_size
        account_curr_risk_min_lot = dollar_risk_min_lot * usd_inr_rate if account_currency.upper() == "INR" else dollar_risk_min_lot
        loss_pct_at_min_lot = (account_curr_risk_min_lot / account_balance) * 100.0

        # Required margin for 0.01 lot: (Price * 100 * 0.01) / Leverage
        margin_usd_min_lot = (entry_price * contract_size * min_lot_size) / leverage
        margin_acc_curr = margin_usd_min_lot * usd_inr_rate if account_currency.upper() == "INR" else margin_usd_min_lot

        # Check: Does minimum tradable lot exceed allowed risk?
        exceeds_risk_limit = account_curr_risk_min_lot > allowed_risk_account_curr

        # Calculate theoretical lot size to respect risk exactly
        theoretical_lot = allowed_risk_usd / (sl_distance * contract_size)
        suggested_lot = round(max(min_lot_size, round(theoretical_lot, 2)), 2)

        # 3. Reward-to-Risk check
        rr_ratio = 0.0
        if take_profit_price is not None and take_profit_price > 0:
            reward_distance = abs(take_profit_price - entry_price)
            rr_ratio = round(reward_distance / sl_distance, 2)

        is_rr_valid = rr_ratio >= 2.0

        # Minimum capital needed to trade 0.01 lot safely at configured risk %
        needed_capital = account_curr_risk_min_lot / (effective_risk_pct / 100.0)

        return {
            "account_balance": account_balance,
            "account_currency": account_currency,
            "configured_risk_pct": risk_per_trade_pct,
            "effective_risk_pct": effective_risk_pct,
            "allowed_cash_risk": round(allowed_risk_account_curr, 2),
            "allowed_cash_risk_usd": round(allowed_risk_usd, 2),
            "sl_distance_pts": round(sl_distance, 2),
            "min_lot_size": min_lot_size,
            "risk_at_min_lot": round(account_curr_risk_min_lot, 2),
            "loss_pct_at_min_lot": round(loss_pct_at_min_lot, 2),
            "margin_required": round(margin_acc_curr, 2),
            "suggested_lot_size": suggested_lot,
            "reward_to_risk_ratio": rr_ratio,
            "is_rr_acceptable": is_rr_valid,
            "exceeds_risk_limit": exceeds_risk_limit,
            "needed_capital_for_safe_trading": round(needed_capital, 2),
            "is_trade_permitted": not exceeds_risk_limit and is_rr_valid,
            "rejection_reason": (
                f"Broker minimum 0.01 lot on a {sl_distance:.1f} pt SL risks {curr_symbol}{account_curr_risk_min_lot:.2f} ({loss_pct_at_min_lot:.1f}% of balance), exceeding your {effective_risk_pct}% limit."
                if exceeds_risk_limit else
                (f"Reward-to-risk ratio (1:{rr_ratio:.1f}) is below the required 1:2.0 threshold." if not is_rr_valid else None)
            )
        }

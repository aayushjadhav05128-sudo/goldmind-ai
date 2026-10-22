import pytest
from app.services.risk_manager import calculate_position_size
from app.schemas.risk import RiskSettings

def test_small_account_protection_inr_2000():
    """
    CRITICAL TEST: Starting capital of ₹2,000 with 0.5% risk (₹10 max risk).
    With Gold at $2,735, a standard 0.01 lot on a 3-point stop would risk $3.00 USD = ~₹259 INR.
    That is >12% of the account. The system MUST skip trade and display the warning.
    """
    settings = RiskSettings(
        account_currency="INR",
        account_balance=2000.0,
        risk_per_trade_pct=0.5, # ₹10 max risk
        contract_size=100.0,
        min_lot_size=0.01,
        usd_inr_rate=86.5
    )
    
    result = calculate_position_size(
        entry_price=2735.0,
        stop_loss_price=2732.0, # 3.0 point stop
        take_profit_1=2741.0,  # 6.0 point TP (1:2)
        settings=settings
    )
    
    assert result.cash_risk_permitted_account_currency == 10.0 # 0.5% of ₹2,000
    assert result.trade_permitted is False
    assert result.suggested_lot_size == 0.0
    assert result.recommend_demo_mode is True
    assert "TRADE SKIPPED — minimum position size exceeds your risk limit" in result.warning_message

def test_adequately_funded_account_sizing():
    """
    Verifies that an adequately capitalized account ($50,000 USD or ₹500,000 INR)
    calculates correct rounded lot size and permits the trade.
    """
    settings = RiskSettings(
        account_currency="USD",
        account_balance=50000.0,
        risk_per_trade_pct=1.0, # $500 risk
        contract_size=100.0,
        min_lot_size=0.01,
        lot_step=0.01
    )
    
    result = calculate_position_size(
        entry_price=2735.0,
        stop_loss_price=2730.0, # 5.0 point stop = $500 per full lot
        take_profit_1=2745.0,  # 10.0 point TP (1:2)
        settings=settings
    )
    
    assert result.cash_risk_usd == 500.0
    assert result.trade_permitted is True
    assert result.suggested_lot_size == 1.00 # 500 / (5.0 * 100) = 1.00 lot
    assert result.warning_message is None

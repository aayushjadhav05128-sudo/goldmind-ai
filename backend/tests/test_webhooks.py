import pytest
from fastapi import HTTPException
from app.services.webhook_service import verify_webhook_secret, check_replay_attack
from app.schemas.webhook import TradingViewAlertPayload
from app.config import settings

def test_webhook_secret_verification():
    # Valid secret
    verify_webhook_secret(settings.TV_WEBHOOK_SECRET)
    
    # Invalid secret should raise 401
    with pytest.raises(HTTPException) as excinfo:
        verify_webhook_secret("wrong_secret_123")
    assert excinfo.value.status_code == 401

def test_webhook_replay_protection():
    payload = TradingViewAlertPayload(
        secret=settings.TV_WEBHOOK_SECRET,
        symbol="XAUUSD",
        timeframe="15",
        timestamp=1761048000000,
        open=2730.0,
        high=2735.0,
        low=2728.0,
        close=2734.0
    )
    
    # First time should succeed and return hash
    h1 = check_replay_attack(payload)
    assert len(h1) == 64
    
    # Second time with identical data within 300s must trigger 409 Conflict
    with pytest.raises(HTTPException) as excinfo:
        check_replay_attack(payload)
    assert excinfo.value.status_code == 409

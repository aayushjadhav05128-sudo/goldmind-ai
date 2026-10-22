from fastapi import APIRouter, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.services.chat_engine import (
    process_chat_message,
    get_conversation_history,
    clear_conversation_history
)

router = APIRouter(prefix="/chat", tags=["AI Chat Analyst"])

class ChatRequest(BaseModel):
    query: str
    symbol: str = "XAUUSD"
    timeframe: str = "15m"
    account_currency: str = "INR"
    account_balance: float = 2000.0
    risk_pct: float = 0.5

@router.post("/message")
async def send_chat_message(req: ChatRequest):
    """
    Submits a natural language trading question to GoldMind AI.
    Pulls real-time candle data, executes full technical + risk calculations,
    and returns a concise institutional recommendation.
    """
    res = await process_chat_message(
        user_query=req.query,
        symbol=req.symbol,
        timeframe=req.timeframe,
        account_currency=req.account_currency,
        account_balance=req.account_balance,
        risk_pct=req.risk_pct
    )
    return res

@router.get("/history")
def get_chat_history():
    """Returns active conversation context."""
    return get_conversation_history()

@router.post("/clear")
def clear_chat():
    """Resets conversational history."""
    clear_conversation_history()
    return {"status": "SUCCESS", "message": "Chat history cleared."}

@router.get("/quick-options")
def get_quick_options():
    """Returns official quick-question shortcut prompts as specified in Section 12."""
    return [
        "Analyse Gold Now",
        "Buy or Sell?",
        "What Patterns Are Forming?",
        "Analyse London/New York Session",
        "Check Liquidity and Market Structure",
        "Compare Current Setup With Historical Setups",
        "Show Model Accuracy and Backtest Results",
        "Why Is This a No-Trade?",
        "Calculate My Risk on ₹2,000"
    ]

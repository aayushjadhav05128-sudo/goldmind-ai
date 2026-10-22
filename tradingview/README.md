# GoldMind AI — TradingView Integration Guide

This guide walks you through connecting your TradingView charts to GoldMind AI using the Pine Script v6 indicator and alert webhook or the manual alert fallback.

---

## 1. Indicator Installation (Pine Script v6)

1. Open [TradingView](https://www.tradingview.com) and chart **XAUUSD** (Gold / US Dollar).
2. Open the **Pine Editor** tab at the bottom of the chart.
3. Click **New** -> Create an empty script.
4. Copy the entire contents of [`goldmind_v6.pine`](file:///tradingview/goldmind_v6.pine) and paste it into the editor.
5. Click **Save** as `GoldMind AI [v6]` and click **Add to chart**.
6. The indicator will plot:
   - **EMA 20** (Yellow line)
   - **EMA 50** (Orange line)
   - **EMA 200** (Purple line)
   - **Confirmed Swing High** (Red circles)
   - **Confirmed Swing Low** (Green circles)
   - **Bullish / Bearish Setup Triggers** (Triangles below/above candles)

---

## 2. Setting Up Automated Webhook Alerts (TradingView Essential / Plus / Premium)

1. In TradingView, right-click on the chart or the indicator and select **Add alert on GoldMind AI...**.
2. **Condition**: Select `GoldMind AI - Market Structure & Signal Alert Engine [v6]`.
3. **Trigger**: Select `Any alert() function call` or `Once Per Bar Close`.
   > **Note**: Bar close is strictly enforced to eliminate repainting.
4. Go to the **Notifications** tab:
   - Check **Webhook URL**.
   - Enter your GoldMind AI backend endpoint:
     `https://your-domain.com/api/v1/webhooks/tradingview` (or `http://localhost:8000/api/v1/webhooks/tradingview` if testing locally with ngrok/tunnel).
5. In the indicator settings (gear icon on the chart), make sure the **Webhook Secret / Bearer Token** matches your backend `.env` setting (`TV_WEBHOOK_SECRET`).
6. Click **Create**.

---

## 3. Alternative: Manual Alert / Free Plan Workflow

If your TradingView account does not support Webhooks (e.g. Basic Free tier):
1. In the alert dialog, set notification to **Send email** or **Show pop-up**.
2. When an alert fires, copy the JSON text from the alert message.
3. Open the GoldMind AI web dashboard -> navigate to **TradingView Integration** tab.
4. Paste the JSON payload into the **Manual Alert Ingestion** box and click **Process Alert**.
5. The backend analyzes the payload, runs multi-timeframe risk validation, and produces the complete trade plan immediately.

---

## 4. Security & Replay Protection

- **Secret Validation**: Every payload includes a shared secret token. Payloads with mismatched secrets are rejected with HTTP 401.
- **Replay Protection**: The backend hashes `(symbol + timeframe + timestamp)` and drops duplicate alerts within a 5-minute window.
- **Rate Limiting**: Limits incoming webhook bursts to prevent denial of service.
- **Zero Account Credentials**: The webhook never asks for broker logins, passwords, or execution keys. GoldMind AI is strictly an analysis assistant.

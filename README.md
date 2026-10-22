# GoldMind AI — AI Forex Trading Analysis Assistant (XAUUSD)

> **Quantitative Decision-Support & Capital Protection Engine**  
> *Engineered for manual execution, mathematical risk control, and multi-timeframe precision.*

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/)

---

## 🚀 GitHub Codespaces Quickstart (Zero Local Setup)

This repository includes a production-ready **`.devcontainer`** environment configured for instant 1-click cloud development in GitHub Codespaces.

### What Happens Automatically in Codespaces:
1. **Container Initialization:** Provisions a Debian Linux environment equipped with Python 3.11 and Node.js 20.
2. **Automated Setup (`.devcontainer/setup.sh`):**
   - Creates the Python virtual environment and installs all backend requirements (`FastAPI`, `scikit-learn`, `joblib`, `pandas`, `pytest`).
   - Installs all Next.js frontend dependencies.
   - Generates initial calibrated ML models (`scripts/train_model.py`).
   - Runs the test suite (`pytest backend/tests`) to ensure 100% test pass.
3. **Automated Server Launch (`.devcontainer/start.sh`):**
   - Automatically starts the FastAPI backend on port **8001**.
   - Automatically starts the Next.js frontend on port **3000**.
4. **Port Forwarding:**
   - **Port 3000:** Opens the GoldMind AI dark trading terminal web interface in your browser.
   - **Port 8001:** Serves the FastAPI REST API and Interactive Swagger Docs at `/docs`.

---

## 1. Core Objective & Philosophy

GoldMind AI is **NOT** an automated trading bot. It does not place orders or connect to broker execution APIs.

Instead, GoldMind AI acts as an institutional trading assistant that:
- Analyzes market structure, Fair Value Gaps (FVG), liquidity sweeps, and trend alignment.
- Issues 4 distinct primary states: **BUY SETUP**, **SELL SETUP**, **WAIT FOR CONFIRMATION**, or **NO TRADE**.
- Calculates exact entry zones, Stop Loss (SL), and Take Profit (TP1, TP2, TP3) levels with a minimum 1:2 Risk-to-Reward ratio.
- Evaluates capital exposure for small accounts (specifically configured for balances as low as **₹2,000 INR**), strictly calculating whether broker lot sizes violate risk tolerance.
- Integrates with **TradingView** via a non-repainting **Pine Script v6** indicator and secure HTTPS webhook (plus a manual ingestion fallback for free plans).
- Grounds all AI explanations in calculated numerical features and timestamps—**never inventing prices or overriding risk parameters**.
- Includes a true **walk-forward backtesting engine** (train, validation, out-of-sample test splits) with conservative same-candle execution and zero look-ahead bias.

---

## 2. Architecture & Tech Stack

```
goldmind-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application & router mounting
│   │   ├── config.py                  # Pydantic BaseSettings & security config
│   │   ├── database.py                # Async SQLAlchemy (SQLite / PostgreSQL)
│   │   ├── models/                    # Signal, Journal, MarketEvent models
│   │   ├── schemas/                   # Pydantic schemas (Candles, Risk, Signals, Webhooks)
│   │   ├── services/
│   │   │   ├── market_data/           # Provider interface (Simulated, CSV, Live)
│   │   │   ├── technical/             # Non-repainting Swings, BOS, CHoCH, FVGs, Indicators
│   │   │   ├── multi_timeframe.py     # 4H, 1H, 15M, 5M Trend Alignment Matrix
│   │   │   ├── risk_manager.py        # Small-account capital protection (₹2,000 INR rule)
│   │   │   ├── signal_engine.py       # Rules-based scoring & hard rejection engine
│   │   │   ├── ai_explainer.py        # Evidence-grounded natural language synthesis
│   │   │   ├── backtester.py          # Chronological bar-by-bar walk-forward tester
│   │   │   └── webhook_service.py     # Replay protection, HMAC secret check, rate limiting
│   │   └── routers/                   # REST API routes
│   ├── tests/                         # Pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx                   # High-performance Dark Trading Terminal UI
│   │   ├── layout.tsx                 # SEO metadata and viewport settings
│   │   └── globals.css                # Custom terminal color tokens & scrollbars
│   ├── package.json
│   └── Dockerfile
├── tradingview/
│   ├── goldmind_v6.pine               # Pine Script v6 Indicator with alert payload
│   └── README.md                      # TradingView setup & alert webhook guide
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 3. Position Sizing & Capital Protection (The ₹2,000 INR Rule)

When trading Gold (**XAUUSD**), instrument contract mechanics are unforgiving on micro-accounts:
- **1.00 Standard Lot** = 100 troy ounces.
- **0.01 Lot** (Broker Minimum) = 1 troy ounce.
- A **$1.00 price move** on a 0.01 lot position equals **$1.00 USD** (= ~**₹86.50 INR**).
- A conservative structural stop loss of **3.5 points** ($3.50) risks **$3.50 USD** (= **₹302.75 INR**).

### What happens on a ₹2,000 INR Account?
If your configured risk limit is **0.5%**, your permitted cash risk is only **₹10.00 INR**.  
Risks on the broker's minimum 0.01 lot would incur **₹302.75 INR**—which is **15.1% of your entire account** on a single trade!

**GoldMind AI refuses to let you blow your account.**  
When `theoretical_lot_size < broker_min_lot`, the engine halts trade execution and displays:

> **"TRADE SKIPPED — minimum position size exceeds your risk limit."**  
> *"Your risk limit is ₹10.00 (0.5%), but broker minimum 0.01 lot on a 3.50 pt SL would risk ₹302.75 (15.1% of balance). Demo trading or a nano/cent account is strongly advised until balance reaches ₹60,550."*

---

## 4. Market-Analysis & Signal Engine

### A. Market Structure
- **Non-Repainting Swings**: Identifies confirmed Swing Highs and Lows using an n-bar lookback window (lagged bar-close confirmation).
- **Break of Structure (BOS)**: Price closes beyond the previous confirmed swing in trend direction.
- **Change of Character (CHoCH)**: Early structural reversal signal where counter-trend close breaches structural pivot.
- **Support & Resistance Clusters**: Automatically clusters historical pivots into high-probability zones with touch counts.

### B. Price Action & Liquidity
- **Fair Value Gaps (FVG)**: Strict 3-candle rule (`Candle 1 High < Candle 3 Low` for Bullish FVG; `Candle 1 Low > Candle 3 High` for Bearish FVG). Tracks mitigation status.
- **Liquidity Sweeps (Stop Hunts)**: Detects upper/lower wicks piercing swing levels that close back inside the range.
- **Equal Highs / Equal Lows (EQH/EQL)**: Identifies liquidity pools resting above double tops or below double bottoms.

### C. Hard Rejection Criteria
A trade setup is **rejected** (forcing `NO TRADE`) if:
1. Market data is stale or candle count < 20 bars.
2. Market spread exceeds 5 pips / points.
3. Market ATR is out of operational bounds ($0.50 to $50.00).
4. Multi-timeframe trend directly contradicts the trigger timeframe.
5. Realistic Risk-to-Reward ratio is less than 1:2.

---

## 5. TradingView Integration

### Automated Webhook Workflow
1. Open **XAUUSD** on TradingView.
2. Load [`tradingview/goldmind_v6.pine`](file:///tradingview/goldmind_v6.pine) in the Pine Editor.
3. Configure an alert on the indicator:
   - **Condition**: `GoldMind AI - Market Structure & Signal Alert Engine [v6]`.
   - **Trigger**: `Once Per Bar Close`.
   - **Webhook URL**: `https://your-domain.com/api/v1/webhooks/tradingview`.
4. In indicator settings, input your secret key (`TV_WEBHOOK_SECRET`).
5. Every confirmed bar close generates a structured JSON payload validated by the backend with replay protection.

### Manual Ingestion Alternative (Free TradingView Accounts)
If your TradingView tier does not include webhooks:
1. Open the GoldMind AI dashboard -> Navigate to the **TradingView & Webhook** tab.
2. Use the **Live Alert Simulator** or paste the alert JSON into the manual ingestion tool.
3. The backend recalculates the full multi-timeframe plan instantaneously.

---

## 6. Walk-Forward Backtesting Engine

- **Strict Chronological Execution**: Bar-by-bar progression guarantees zero lookahead bias.
- **Conservative Same-Candle Resolution**: If both High reaches TP and Low breaches SL within the same bar, the simulator assumes the Stop Loss was hit first (worst-case assumption).
- **Out-of-Sample Partitioning**:
  - **In-Sample (Train 60%)**: Strategy calibration.
  - **Validation (20%)**: Tuning checks.
  - **Out-of-Sample (Test 20%)**: True forward out-of-sample evaluation.
- Reports **Expectancy in R**, **Profit Factor**, **Max Drawdown**, and **Win Rate**.

---

## 7. Running GoldMind AI Locally

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Backend API will be running at: `http://localhost:8000`  
Interactive Swagger Docs at: `http://localhost:8000/docs`

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend terminal will be running at: `http://localhost:3000`

### 3. Running Automated Tests
```bash
cd backend
pytest -v
```

---

## 8. Docker Deployment

To launch the full stack in isolated containers:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

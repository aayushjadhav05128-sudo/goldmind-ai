#!/usr/bin/env bash

echo "=========================================="
echo " Starting GoldMind AI Services"
echo "=========================================="

REPO_ROOT="$(pwd)"

# Ensure environment file exists
if [ ! -f "$REPO_ROOT/backend/.env" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/backend/.env"
fi

# Ensure log dir exists
mkdir -p /tmp/goldmind-logs

# Kill any existing processes bound to 8001 or 3000
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true

# 1. Start FastAPI Backend in background
echo "--> Launching FastAPI Backend on port 8001..."
"$REPO_ROOT/backend/venv/bin/python" -m uvicorn app.main:app \
  --app-dir "$REPO_ROOT/backend" \
  --host 0.0.0.0 \
  --port 8001 > /tmp/goldmind-logs/backend.log 2>&1 &

sleep 2

# 2. Start Next.js Frontend in background
echo "--> Launching Next.js Frontend on port 3000..."
cd "$REPO_ROOT/frontend"
npm run dev -- -p 3000 -H 0.0.0.0 > /tmp/goldmind-logs/frontend.log 2>&1 &
cd "$REPO_ROOT"

echo "=========================================="
echo " GoldMind AI is running in your Codespace!"
echo " Frontend Web UI:  http://localhost:3000"
echo " Backend Swagger:  http://localhost:8001/docs"
echo " Live Logs:        tail -f /tmp/goldmind-logs/*.log"
echo "=========================================="

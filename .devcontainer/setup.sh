#!/usr/bin/env bash
set -e

echo "=========================================="
echo " Setting up GoldMind AI Codespace"
echo "=========================================="

REPO_ROOT="$(pwd)"

# 1. Environment file setup
if [ ! -f "$REPO_ROOT/backend/.env" ]; then
  echo "--> Initializing backend/.env from .env.example..."
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/backend/.env"
fi

# 2. Python Virtual Environment
echo "--> Creating Python virtual environment in backend/venv..."
python3 -m venv "$REPO_ROOT/backend/venv"
source "$REPO_ROOT/backend/venv/bin/activate"

echo "--> Upgrading pip..."
pip install --upgrade pip

echo "--> Installing backend Python dependencies..."
pip install -r "$REPO_ROOT/backend/requirements.txt"

# 3. Frontend Node.js dependencies
echo "--> Installing frontend dependencies..."
cd "$REPO_ROOT/frontend"
npm install
cd "$REPO_ROOT"

# 4. Train & Approve ML Model bundle (if not already trained)
echo "--> Verifying / Training production machine learning model..."
python3 "$REPO_ROOT/backend/scripts/train_model.py" --bars 1200 --timeframe 15m --target label_triple_barrier_buy

# 5. Run automated test suite
echo "--> Running backend test suite to verify installation..."
pytest "$REPO_ROOT/backend/tests" -v

echo "=========================================="
echo " GoldMind AI Codespace Setup Complete!"
echo "=========================================="

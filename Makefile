.PHONY: setup dev backend frontend test train backtest

setup:
	bash .devcontainer/setup.sh

dev:
	bash .devcontainer/start.sh

backend:
	cd backend && ./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

frontend:
	cd frontend && npm run dev -- -p 3000

test:
	./backend/venv/bin/pytest backend/tests -v

train:
	./backend/venv/bin/python backend/scripts/train_model.py --bars 1200 --timeframe 15m --target label_triple_barrier_buy

backtest:
	./backend/venv/bin/python backend/scripts/run_backtest.py --bars 1200 --timeframe 15m --strategy STRAT-XAU-001

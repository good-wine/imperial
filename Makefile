.PHONY: install lint format typecheck security test coverage eval run-api run-worker run-dashboard docker-up clean

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV) --without-pip
	curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/opencode/get-pip.py
	$(VENV)/bin/python /tmp/opencode/get-pip.py
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src ingestion security dashboard scripts main.py worker.py tests

format:
	$(PY) -m ruff format src ingestion security dashboard scripts main.py worker.py tests

format-check:
	$(PY) -m ruff format --check src ingestion security dashboard scripts main.py worker.py tests

typecheck:
	$(PY) -m mypy src ingestion security main.py worker.py

security:
	$(PY) -m bandit -c .bandit -r src ingestion security dashboard scripts main.py worker.py

test:
	$(PY) -m pytest tests/ -q -p no:cacheprovider

coverage:
	$(PY) -m pytest tests/ -q -p no:cacheprovider --cov --cov-report=term-missing --cov-fail-under=55

eval:
	$(PY) scripts/eval_metrics.py

# Local single-node run (SQLite + memory vector store + eager worker).
run-api:
	$(PY) -m uvicorn main:app --host 127.0.0.1 --port 8001

run-worker:
	$(PY) -m celery -A worker.app worker --loglevel=info --concurrency=2

run-dashboard:
	API_BASE=http://127.0.0.1:8001 $(PY) -m streamlit run dashboard/app.py \
		--server.port=8502 --server.address=127.0.0.1 --server.headless=true

docker-up:
	docker compose --profile full up --build -d

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml

sbom:
	bash scripts/sbom.sh

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock

migrate:
	$(PY) -m alembic upgrade head

# Vulnerability audit. PYSEC-2026-3740 (nltk, unpatched upstream) is waived:
# NLTK model-persistence APIs are unreachable from our code paths (no
# caller-controlled model paths); re-audit on every nltk upgrade.
audit:
	$(PY) -m pip-audit --desc --ignore-vuln PYSEC-2026-3740

# Full quality gate, same order as CI.
gate: format-check lint typecheck security test

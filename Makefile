.PHONY: help install run test evals evals-safety judge clean

help:
	@echo "make install      create .venv and install dependencies"
	@echo "make run          start the chat CLI"
	@echo "make test         unit + integration tests (no API key needed)"
	@echo "make evals        golden-set evaluation suites (no API key needed)"
	@echo "make evals-safety adversarial suite only"
	@echo "make judge        evals with the live model and the LLM judge"
	@echo "make clean        remove generated runtime state"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

run:
	PYTHONPATH=src .venv/bin/python -m agent

test:
	PYTHONPATH=src:tests .venv/bin/python -m pytest tests/ -q

evals:
	PYTHONPATH=src:tests .venv/bin/python evals/run_evals.py

evals-safety:
	PYTHONPATH=src:tests .venv/bin/python evals/run_evals.py --suite safety

judge:
	PYTHONPATH=src:tests .venv/bin/python evals/run_evals.py --live --judge

clean:
	rm -rf .runtime evals/results
	rm -f data/golden_bucket/candidates/*.json data/golden_bucket/.embedding_cache.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

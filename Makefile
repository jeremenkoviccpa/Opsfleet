.PHONY: help install run test evals evals-safety judge diagrams clean

help:
	@echo "make install      create .venv and install dependencies"
	@echo "make run          start the chat CLI"
	@echo "make test         unit + integration tests (no API key needed)"
	@echo "make evals        golden-set evaluation suites (no API key needed)"
	@echo "make evals-safety adversarial suite only"
	@echo "make judge        evals with the live model and the LLM judge"
	@echo "make diagrams     re-render docs/diagrams/*.png from the Mermaid source"
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

diagrams:
	@command -v npx >/dev/null || { echo "needs node/npx"; exit 1; }
	@mkdir -p docs/diagrams
	@python3 - <<'PY'
	import pathlib, re, subprocess, tempfile
	names = {1:"1-system-architecture",2:"2-compute-and-request-path",3:"3-where-the-data-lives",
	         4:"4-agent-graph",5:"5-anatomy-of-a-turn",6:"6-safety-enforcement-points"}
	src = pathlib.Path("docs/ARCHITECTURE.md").read_text()
	for i, b in enumerate(re.findall(r"```mermaid\n(.*?)```", src, re.DOTALL), 1):
	    with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as fh:
	        fh.write(b); path = fh.name
	    out = f"docs/diagrams/{names[i]}.png"
	    subprocess.run(["npx","-y","@mermaid-js/mermaid-cli","-i",path,"-o",out,"-t","dark","-s","3"], check=True)
	    print("rendered", out)
	PY

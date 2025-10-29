# Repository Guidelines

## Project Structure & Module Organization
- `aios/`: Core agent runtime (LLM clients, memory, scheduler, storage, utils).
- `pyopenagi/`: Agent logic, tools, queues, and shared utils.
- `scripts/`: Entry points for experiments (e.g., `agent_attack.py`).
- `config/`: YAML configs per scenario (DPI, OPI, MP, mixed, PoT).
- `runtime/`: Local execution server and helpers.
- `data/`: JSONL datasets for tools/tasks; do not commit generated PII.
- `memory_db/`, `memory_defense/`: Persistence and defense utilities.
- `images/`, `README.md`, `Dockerfile`, `requirements*.txt`.

## Build, Test, and Development Commands
```bash
# Setup (Python 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt  # includes ruff, pytest, pre-commit

# Lint/format
ruff check .
ruff format .

# Run examples
python scripts/agent_attack.py --cfg_path config/DPI.yml
python scripts/agent_attack_pot.py

# Optional: pre-commit hooks
pre-commit install && pre-commit run -a
```

## Coding Style & Naming Conventions
- Python: 4-space indent, type hints for public APIs, concise docstrings.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Keep new agents under `pyopenagi/agents/` and tools under `pyopenagi/tools/`.
- Use ruff for linting/formatting; fix warnings before PRs.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` mirroring package paths (e.g., `tests/pyopenagi/tools/test_google_search.py`).
- Name tests `test_*.py`; prefer fast, isolated tests; mock network calls and API clients.
- Run with `pytest -q`; add targeted runs via `pytest -k name`.

## Commit & Pull Request Guidelines
- Commits: short, imperative summaries (e.g., "Add DPI config loader"). Current history favors concise messages; Conventional Commits welcome but not required.
- PRs: include purpose, key changes, how to run (commands + config), and any security implications. Link issues; add screenshots/logs when relevant.
- CI friendliness: ensure `ruff check`, `ruff format --check`, and `pytest` pass locally.

## Security & Configuration Tips
- Set keys via env vars: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `GPLACES_API_KEY`, `RAPID_API_KEY`. For local models, run `ollama serve` and configure model names in YAML.
- Do not hardcode credentials or commit artifacts from `memory_db/`. Redact sensitive data in examples.

## Extending the Bench
- Add scenarios by creating a YAML in `config/` and, if needed, new tools/agents under `pyopenagi/`. Keep CLI entry points in `scripts/` and document usage in `README.md`.

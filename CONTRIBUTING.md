# Contributing to knowledge-pipeline

Thanks for your interest! This project values simplicity and practicality over complexity.

## Principles

1. **Zero frameworks** — We use Python stdlib wherever possible. Don't add LangChain, LlamaIndex, or similar.
2. **Each layer is independent** — Changes to one layer shouldn't break others.
3. **Deterministic pipeline** — The pipeline should produce consistent, reproducible results.
4. **Ollama-first** — The default config should work with a local Ollama install, no API keys needed.

## Getting started

```bash
git clone https://github.com/MakiDevelop/knowledge-pipeline.git
cd knowledge-pipeline
pip install -r requirements.txt
cp .env.example .env

# Run tests
pytest tests/ -v
```

## Status

This repo is a **public extract** of a private personal system (`mk-brain`).  
Daily development happens in mk-brain and is **not** merged back here.

Please do **not** open PRs that add mk-brain-only features (RSS ingestion, Ghost publishing, dashboard, Docker one-command stacks, extra ingest sources) in order to "catch up." Those belong in the private line, if anywhere.

## What we'd love help with

Useful contributions for **this snapshot**:

- Bugfixes and failing-test reproductions
- Documentation and examples that match the code that is actually here
- HTML extraction edge cases in `enrich.py`
- Search quality improvements that stay inside the current layers

Not in scope: feature parity with mk-brain, new ingest sources, Compose/one-command product packaging.

## Code style

- Python 3.12+
- Use `ruff` for linting: `ruff check .`
- Type hints are welcome but not required
- Comments only where logic isn't self-evident
- Prefer stdlib over external packages

## Pull request process

1. Fork the repo and create a feature branch
2. Make your changes
3. Add/update tests if applicable
4. Run `ruff check .` and fix any issues
5. Submit a PR with a clear description of what and why

## Scoring prompt changes

Changes to the scoring prompt in `score.py` affect all downstream results. If you modify it:

1. Run scoring on a small sample (`python3 score.py --limit 10`)
2. Compare scores before/after
3. Document why the change improves quality

## Questions?

Open an issue! We're friendly.

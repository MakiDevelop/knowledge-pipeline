# Contributing to knowledge-pipeline

Thanks for your interest! This project values simplicity and practicality over complexity.

## Principles

1. **Zero frameworks** — We use Python stdlib wherever possible. Don't add LangChain, LlamaIndex, or similar.
2. **Each layer is independent** — Changes to one layer shouldn't break others.
3. **Deterministic pipeline** — The pipeline should produce consistent, reproducible results.
4. **Ollama-first** — The default config should work with a local Ollama install, no API keys needed.

## Getting started

```bash
git clone https://github.com/makifordevelop/knowledge-pipeline.git
cd knowledge-pipeline
pip install -r requirements.txt
cp .env.example .env

# Run tests
pytest tests/ -v
```

## What we'd love help with

### Good first issues

- **Add RSS ingestion** — Extend `ingest.py` to accept RSS feed URLs and auto-extract article links
- **Add Obsidian ingestion** — Read URLs from Obsidian vault markdown files
- **Improve HTML extraction** — The `_HTMLTextExtractor` in `enrich.py` is basic; handle more edge cases
- **Add `--output csv` to search.py** — Export search results as CSV
- **Add scoring dimension visualization** — A simple radar chart showing the 8 dimensions

### Medium

- **Alternative embedding models** — Support for OpenAI embeddings, Cohere, etc.
- **Batch scoring with async** — Score multiple items concurrently
- **Web UI for search** — Simple HTML page served alongside the API
- **Docker compose** — One-command setup with Ollama + pipeline

### Advanced

- **Custom scoring dimensions** — Let users define their own scoring criteria via config
- **Knowledge consolidation** — Cross-topic synthesis (grouping related items and generating insights)
- **Webhook ingestion** — HTTP endpoint to receive URLs from browsers, Slack, Discord
- **MCP server implementation** — Full MCP protocol support for Claude Desktop

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

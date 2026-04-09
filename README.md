# knowledge-pipeline

**Auto-triage your knowledge. Score, route, search — no frameworks.**

[繁體中文 README](README.zh-TW.md)

Most bookmark managers are graveyards. You save 500 URLs and never look at them again.

`knowledge-pipeline` is different. It's a **6-layer deterministic pipeline** that automatically:

1. **Ingests** URLs from any source (CLI, file, API)
2. **Enriches** them with full-text extraction and LLM-generated summaries
3. **Scores** each item on 8 dimensions using an LLM — not just "relevant or not", but *how* valuable, *how* novel, *how* actionable
4. **Routes** each item to a destination: write about it, research deeper, fact-check it, act on it, or archive it
5. **Embeds** everything with dense + sparse vectors for hybrid semantic search
6. **Serves** a search API that any AI agent can query

```
URL in → Fetch → Score → Route → Embed → Search API out
                   ↓
          signal=82, route=writer
          "This paper introduces a novel framework for..."
```

## Why this exists

If you use multiple AI agents (Claude, ChatGPT, Gemini, Copilot, local models...), your knowledge is scattered across dozens of context windows that vanish when the session ends.

This pipeline gives your agents a **shared, persistent, scored knowledge layer**. Instead of each agent starting from zero, they can query what you've already collected — and the pipeline has already decided what's worth their attention.

## What makes this different

| Feature | Typical RAG | knowledge-pipeline |
|---------|-------------|-------------------|
| Knowledge flow | Passive (you ask, it answers) | **Active** (auto-score, auto-route) |
| Quality signal | None (all items equal) | **8-dimension LLM scoring + signal score** |
| Routing | None | **Auto-routes to: write / research / validate / act / archive** |
| Framework | LangChain, LlamaIndex, etc. | **Zero frameworks. Pure Python stdlib + numpy** |
| LLM backend | Usually OpenAI-only | **Any OpenAI-compatible API (Ollama, OpenAI, Anthropic...)** |
| Search | Dense-only | **Hybrid: 70% dense + 30% sparse + optional reranking** |

## Quickstart

### Prerequisites

- Python 3.12+
- An OpenAI-compatible LLM (local [Ollama](https://ollama.ai) recommended — no API key needed)

### Install

```bash
git clone https://github.com/MakiDevelop/knowledge-pipeline.git
cd knowledge-pipeline
pip install -r requirements.txt

# Start Ollama with a model (if using local LLM)
ollama pull qwen2.5:7b
```

### Configure

```bash
cp .env.example .env
# Edit .env if you want to use OpenAI or a different model
```

### Run the pipeline

```bash
# 1. Ingest URLs
python3 ingest.py https://arxiv.org/abs/2401.12345 https://simonwillison.net/2024/...

# Or from a file (one URL per line)
python3 ingest.py urls.txt

# 2. Enrich (fetch + summarize)
python3 enrich.py

# 3. Score (8-dimension analysis + routing)
python3 score.py

# 4. Embed (dense + sparse vectors)
python3 embed.py

# 5. Search
python3 search.py "AI agent orchestration"
python3 search.py "knowledge management" --rerank

# 6. Serve (HTTP API for your agents)
python3 serve.py
# → http://localhost:8780/search?q=AI+agents
```

## Scoring dimensions

Every item is scored on 8 dimensions (0-5 each) by an LLM:

| Dimension | What it measures |
|-----------|-----------------|
| `knowledge_density` | How structured and information-rich |
| `novelty` | New perspectives or counter-intuitive insights |
| `evidence_strength` | Supported by data, examples, or logic |
| `actionability` | Can you act on this immediately |
| `risk_level` | Technical or societal risk involved |
| `time_horizon` | Impact timeframe: short / mid / long |
| `emotional_noise` | Emotion vs substance (penalty) |
| `source_credibility` | Trustworthiness of the source |

These combine into a **signal score (0-100)** and a **route**:

| Route | Meaning | Example trigger |
|-------|---------|-----------------|
| `writer` | Good for publishing/writing | High density + strong evidence |
| `research` | Needs deeper investigation | High novelty, needs more evidence |
| `action` | Directly actionable | High actionability, low risk |
| `validator` | Needs fact-checking | High risk or high emotion + low evidence |
| `archive` | Low priority, file away | Default |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  knowledge.db (SQLite)            │
├──────────────────────────────────────────────────┤
│                                                  │
│  ingest.py ─→ enrich.py ─→ score.py ─→ embed.py │
│     (L1)        (L2)         (L3)        (L4)    │
│                                                  │
│  search.py ←── serve.py ←── Your AI Agents       │
│     (L5)        (L6)         (Claude, GPT, ...)  │
│                                                  │
└──────────────────────────────────────────────────┘
```

Each layer is independent. You can:
- Run them separately or chain them
- Replace any layer without touching others
- Skip layers you don't need (e.g., skip embedding if you only want scoring)

## Using with AI agents

### As an MCP tool
Point your Claude/GPT/etc. MCP config to the serve.py endpoint:
```json
{
  "tools": [{
    "name": "search_knowledge",
    "url": "http://localhost:8780/search",
    "params": {"q": "query", "k": 10}
  }]
}
```

### As a function call
```python
import urllib.request, json
resp = urllib.request.urlopen("http://localhost:8780/search?q=AI+agents&k=5")
results = json.loads(resp.read())
for r in results["results"]:
    print(f"[{r['signal_score']}] {r['title']}")
```

## Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.12+ | Simplicity, stdlib-first |
| Database | SQLite | Zero setup, portable, surprisingly fast |
| Embedding | BAAI/bge-m3 | Best multilingual dense+sparse in one model |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder for precision |
| LLM | Any OpenAI-compatible | Ollama (local, free), OpenAI, Anthropic... |
| Web server | stdlib HTTPServer | Zero dependencies for serving |

## Requirements

```
numpy
FlagEmbedding
```

That's it. Two packages beyond stdlib.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

We especially welcome:
- New scoring dimensions or routing strategies
- Alternative LLM backends or prompts
- Ingestion from new sources (RSS, Slack, Discord, Obsidian...)
- Search quality improvements
- Documentation and examples

## Origin

This project is distilled from [mk-brain](https://github.com/makifordevelop/mk-brain), a personal knowledge infrastructure running 1,600+ items across 6 layers. The scoring and routing system has been refined through months of daily use.

## License

MIT

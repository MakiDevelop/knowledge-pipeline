# knowledge-pipeline

**自動分揀你的知識。評分、路由、搜尋——零框架。**

大多數書籤管理工具都是墳場。你存了 500 個 URL,然後再也沒有打開過。

`knowledge-pipeline` 不一樣。它是一條**6 層確定性管道**,自動:

1. **匯入** 來自任何來源的 URL(CLI、檔案、API)
2. **豐富** 內容——全文抓取 + LLM 生成的摘要
3. **評分** 每一則項目 8 個維度——不只是「相關不相關」,而是「有多有價值、有多新穎、有多可行動」
4. **路由** 每一則項目到目的地:寫文章、深入研究、事實查核、立即行動、或歸檔
5. **嵌入** 一切,用 dense + sparse 向量做混合語意搜尋
6. **提供** 搜尋 API,任何 AI Agent 都能查

```
URL 進 → 抓取 → 評分 → 路由 → 嵌入 → 搜尋 API 出
                  ↓
         signal=82, route=writer
         「這篇論文介紹了一個全新的框架...」
```

[English README](README.md)

## 為什麼存在

如果你使用多個 AI Agent(Claude、ChatGPT、Gemini、Copilot、地端模型...),你的知識散落在幾十個 context window 裡,session 結束那一刻就消失了。

這條 pipeline 給你的 Agent 們一個**共享的、持久的、有評分的知識層**。每個 Agent 不再從零開始,它們可以查詢你已經收集的東西——而且管道已經決定了哪些值得它們花時間。

## 有什麼不一樣

| 特色 | 典型 RAG | knowledge-pipeline |
|------|---------|-------------------|
| 知識流向 | 被動(你問它答) | **主動(自動評分、自動路由)** |
| 品質訊號 | 無(所有項目平等) | **8 維 LLM 評分 + signal score** |
| 路由 | 無 | **自動路由到:寫作 / 研究 / 驗證 / 行動 / 歸檔** |
| 框架 | LangChain、LlamaIndex 等 | **零框架。純 Python stdlib + numpy** |
| LLM 後端 | 通常只有 OpenAI | **任何 OpenAI-compatible API(Ollama、OpenAI、Anthropic...)** |
| 搜尋 | 只有 dense | **混合:70% dense + 30% sparse + 可選 rerank** |

## 快速開始

### 前置需求

- Python 3.12+
- OpenAI-compatible LLM(推薦本地 [Ollama](https://ollama.ai)——不需要 API key)

### 安裝

```bash
git clone https://github.com/MakiDevelop/knowledge-pipeline.git
cd knowledge-pipeline
pip install -r requirements.txt

# 啟動 Ollama 並下載模型(如果使用本地 LLM)
ollama pull qwen2.5:7b
```

### 30 秒體驗(不需要 LLM)

```bash
bash quickstart.sh
```

會載入預評分的示範資料,讓你立刻看到評分和路由的效果。

### 設定

```bash
cp .env.example .env
# 如果要用 OpenAI 或其他模型,編輯 .env
```

### 完整 pipeline

```bash
# 1. 匯入 URL
python3 ingest.py https://arxiv.org/abs/2401.12345 https://simonwillison.net/2024/...

# 或從檔案(每行一個 URL)
python3 ingest.py urls.txt

# 2. 豐富(抓取 + 摘要)
python3 enrich.py

# 3. 評分(8 維分析 + 路由)
python3 score.py

# 4. 嵌入(dense + sparse 向量)
python3 embed.py

# 5. 搜尋
python3 search.py "AI agent orchestration"
python3 search.py "knowledge management" --rerank

# 6. 提供 API(給你的 Agent 用)
python3 serve.py
# → http://localhost:8780/search?q=AI+agents
```

## 評分維度

每一則項目會被 LLM 在 8 個維度上評分(各 0-5):

| 維度 | 衡量什麼 |
|------|---------|
| `knowledge_density` | 資訊密度,是否結構化 |
| `novelty` | 新穎度,是否提出新觀點 |
| `evidence_strength` | 證據強度,有數據還是純猜測 |
| `actionability` | 可行動性,讀完能否立即行動 |
| `risk_level` | 風險等級,技術/社會風險 |
| `time_horizon` | 影響時間:短期 / 中期 / 長期 |
| `emotional_noise` | 情緒噪音(扣分項) |
| `source_credibility` | 來源可信度 |

這些會合成一個 **signal score(0-100)** 和一個**路由**:

| 路由 | 意義 | 觸發範例 |
|------|------|---------|
| `writer` | 適合寫作/發佈 | 高密度 + 強證據 |
| `research` | 需要深入調查 | 高新穎度,需要更多證據 |
| `action` | 直接可行動 | 高可行動性,低風險 |
| `validator` | 需要事實查核 | 高風險或高情緒 + 低證據 |
| `archive` | 低優先級,歸檔 | 預設 |

## 架構

```
┌──────────────────────────────────────────────────┐
│                  knowledge.db (SQLite)            │
├──────────────────────────────────────────────────┤
│                                                  │
│  ingest.py ─→ enrich.py ─→ score.py ─→ embed.py │
│     (L1)        (L2)         (L3)        (L4)    │
│                                                  │
│  search.py ←── serve.py ←── 你的 AI Agent       │
│     (L5)        (L6)         (Claude, GPT, ...)  │
│                                                  │
└──────────────────────────────────────────────────┘
```

每一層都獨立。你可以:
- 分開跑,或串起來跑
- 替換任何一層而不影響其他層
- 跳過不需要的層(例如只用評分不用嵌入)

## 搭配 AI Agent 使用

### 作為 MCP 工具
把你的 Claude/GPT/等等 MCP 設定指向 serve.py endpoint:
```json
{
  "tools": [{
    "name": "search_knowledge",
    "url": "http://localhost:8780/search",
    "params": {"q": "query", "k": 10}
  }]
}
```

### 作為 function call
```python
import urllib.request, json
resp = urllib.request.urlopen("http://localhost:8780/search?q=AI+agents&k=5")
results = json.loads(resp.read())
for r in results["results"]:
    print(f"[{r['signal_score']}] {r['title']}")
```

## 技術棧

| 元件 | 選擇 | 為什麼 |
|------|------|-------|
| 語言 | Python 3.12+ | 簡單、stdlib 優先 |
| 資料庫 | SQLite | 零設定、可攜、意外地快 |
| 嵌入 | BAAI/bge-m3 | 最佳的多語言 dense+sparse 單一模型 |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder 提升精確度 |
| LLM | 任何 OpenAI-compatible | Ollama(本地免費)、OpenAI、Anthropic... |
| Web server | stdlib HTTPServer | 零依賴 |

## 需求

```
numpy
FlagEmbedding
```

就這樣。stdlib 以外只有兩個套件。

## 貢獻

請見 [CONTRIBUTING.md](CONTRIBUTING.md)。

特別歡迎:
- 新的評分維度或路由策略
- 其他 LLM 後端或 prompt
- 從新來源匯入(RSS、Slack、Discord、Obsidian...)
- 搜尋品質改善
- 文件和範例

## 起源

這個專案從 [mk-brain](https://github.com/MakiDevelop/mk-brain) 提煉而來,那是一個運行中的個人知識基礎設施,跨 6 層管理 1,600+ 則項目。評分和路由系統經過數個月每日使用的精煉。

## License

MIT

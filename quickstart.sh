#!/bin/bash
# quickstart.sh — Try knowledge-pipeline in 30 seconds (no LLM needed)
#
# This loads pre-scored sample data so you can explore the pipeline
# without setting up Ollama or any LLM backend.

set -e

echo "=== knowledge-pipeline quickstart ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# Load sample data
echo "1. Loading sample data..."
python3 examples/load_sample.py
echo ""

# Show scored items
echo "2. Items by signal score:"
python3 -c "
from config import get_db_connection
conn = get_db_connection()
rows = conn.execute('SELECT title, signal_score, route_to FROM items ORDER BY signal_score DESC').fetchall()
for r in rows:
    print(f'  [{r[\"signal_score\"]:3d}] [{r[\"route_to\"]:9s}] {r[\"title\"]}')
conn.close()
"
echo ""

# Show route distribution
echo "3. Route distribution:"
python3 -c "
from config import get_db_connection
conn = get_db_connection()
rows = conn.execute('SELECT route_to, COUNT(*) as cnt FROM items GROUP BY route_to ORDER BY cnt DESC').fetchall()
for r in rows:
    print(f'  {r[\"route_to\"]:12s} {r[\"cnt\"]} items')
conn.close()
"
echo ""

# Show score distribution
echo "4. Score statistics:"
python3 -c "
from config import get_db_connection
conn = get_db_connection()
r = conn.execute('SELECT MIN(signal_score) as lo, MAX(signal_score) as hi, AVG(signal_score) as avg, COUNT(*) as n FROM items').fetchone()
print(f'  {r[\"n\"]} items | min={r[\"lo\"]} max={r[\"hi\"]} avg={r[\"avg\"]:.0f}')
conn.close()
"
echo ""

echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  # Ingest your own URLs:"
echo "  python3 ingest.py https://example.com/article"
echo ""
echo "  # With an LLM (Ollama):"
echo "  ollama pull qwen2.5:7b"
echo "  python3 enrich.py    # Fetch + summarize"
echo "  python3 score.py     # 8-dimension scoring"
echo ""
echo "  # With embeddings (requires: pip install FlagEmbedding numpy):"
echo "  python3 embed.py     # Generate vectors"
echo "  python3 search.py 'AI agents'  # Semantic search"
echo ""
echo "  # Serve API for your AI agents:"
echo "  python3 serve.py     # http://localhost:8780/search?q=..."

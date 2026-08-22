#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================================="
echo " Starting Video Vector Search Application (FastAPI + UI) "
echo "=========================================================="

cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

source .venv/bin/activate

echo "Backend starting on http://localhost:8000"
echo "Interactive Swagger API Docs: http://localhost:8000/docs"
echo "Web UI (User & Admin Facets): http://localhost:8000"
echo ""

uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

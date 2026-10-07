#!/usr/bin/env bash
# ==============================================================================
# Docker Launch Script for Milvus RAG Stack
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo ">>> [Docker] Initializing required volume and data directories..."
mkdir -p volumes/etcd volumes/minio volumes/milvus volumes/model_cache data/raw data/processed data/embeddings data/eval

# Ensure .env file exists
if [ ! -f .env ]; then
    echo ">>> .env not found, copying from .env.example..."
    cp .env.example .env
fi

ACTION="${1:-up}"

if [ "${ACTION}" = "up" ]; then
    echo ">>> Starting Milvus RAG Stack with Docker Compose..."
    docker compose -f docker-compose.yml up -d
    echo ">>> Milvus Standalone running at localhost:19530"
    echo ">>> Attu UI running at http://localhost:3000"
    echo ">>> Streamlit UI running at http://localhost:8502"
    echo ">>> RAG FastAPI running at http://localhost:8000/docs"

elif [ "${ACTION}" = "down" ]; then
    echo ">>> Stopping Milvus RAG Stack..."
    docker compose -f docker-compose.yml down

elif [ "${ACTION}" = "logs" ]; then
    docker compose -f docker-compose.yml logs -f
else
    echo "Usage: $0 {up|down|logs}"
fi

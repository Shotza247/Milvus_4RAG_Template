#!/usr/bin/env bash
# ==============================================================================
# Podman Launch Script for Milvus RAG Stack
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo ">>> [Podman] Initializing required volume and data directories..."
mkdir -p volumes/etcd volumes/minio volumes/milvus volumes/model_cache data/raw data/processed data/embeddings data/eval

# Ensure .env file exists
if [ ! -f .env ]; then
    echo ">>> .env not found, copying from .env.example..."
    cp .env.example .env
fi

ACTION="${1:-up}"

if [ "${ACTION}" = "up" ]; then
    echo ">>> Starting Milvus RAG Stack with Podman Compose..."
    if command -v podman-compose >/dev/null 2>&1; then
        podman-compose -f compose.yaml up -d
    elif podman compose version >/dev/null 2>&1; then
        podman compose -f compose.yaml up -d
    else
        echo "ERROR: Neither 'podman compose' nor 'podman-compose' was found."
        exit 1
    fi
    echo ">>> Milvus Standalone running at localhost:19530"
    echo ">>> Attu UI running at http://localhost:3000"
    echo ">>> Streamlit UI running at http://localhost:8502"
    echo ">>> RAG FastAPI running at http://localhost:8000/docs"

elif [ "${ACTION}" = "down" ]; then
    echo ">>> Stopping Milvus RAG Stack..."
    if command -v podman-compose >/dev/null 2>&1; then
        podman-compose -f compose.yaml down
    else
        podman compose -f compose.yaml down
    fi

elif [ "${ACTION}" = "logs" ]; then
    if command -v podman-compose >/dev/null 2>&1; then
        podman-compose -f compose.yaml logs -f
    else
        podman compose -f compose.yaml logs -f
    fi
else
    echo "Usage: $0 {up|down|logs}"
fi

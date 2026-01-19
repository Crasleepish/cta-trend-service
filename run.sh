#!/usr/bin/env bash
set -euo pipefail

exec uv run uvicorn src.main:app "$@"

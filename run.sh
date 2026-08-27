#!/usr/bin/env bash
# Sobe a demo em http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q -r requirements.txt
exec ./.venv/bin/uvicorn app.main:app --reload --port "${PORT:-8000}"

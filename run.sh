#!/usr/bin/env bash
# Starts backend (port 8000) and frontend (port 5173), opens the browser.
cd "$HOME/dfpe/backend" && source .venv/bin/activate
uvicorn app.main:app --port 8000 &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd "$HOME/dfpe/frontend"
(sleep 4; open http://localhost:5173) &
npm run dev

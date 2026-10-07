#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting RAG Note..."

# Backend (no --reload: it would restart whenever models/data files change)
cd "$ROOT/backend"
[ -f .env ] || cp .env.example .env
python3 main.py &
BACKEND_PID=$!

# Frontend
cd "$ROOT/frontend"
[ -d node_modules ] || npm install
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000   (first start downloads the model once)"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop."

trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait

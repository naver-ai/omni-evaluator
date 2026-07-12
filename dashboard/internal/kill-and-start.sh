#!/usr/bin/env bash
# Free ports and start FastAPI
set -e
DASHBOARD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT_INTERNAL=11592

echo "=== Killing processes on ${PORT_INTERNAL} ==="
pids=$(lsof -t -i :$PORT_INTERNAL 2>/dev/null || true)
if [ -n "$pids" ]; then
  echo "  ${PORT_INTERNAL}: kill -9 $pids"
  kill -9 $pids 2>/dev/null || true
fi
sleep 3
echo "=== Starting FastAPI on ${PORT_INTERNAL} ==="
cd "$DASHBOARD_ROOT"
"$DASHBOARD_ROOT/internal/run-backend.sh"

#!/usr/bin/env bash
# Lives on your Mac (e.g. ~/bin/carla_view or ~/carla_view.sh) — not in the research repo.
# Usage: ./carla_view.sh
set -euo pipefail

HOST="${LAMBDA_HOST:-170.9.12.5}"
USER="${LAMBDA_USER:-ubuntu}"
KEY="${LAMBDA_KEY:-$HOME/.ssh/id_ed25519}"
PORT="${STREAM_PORT:-8080}"

PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
[[ -n "${PIDS:-}" ]] && kill $PIDS 2>/dev/null || true
pkill -f "ssh.*-L ${PORT}:127.0.0.1:${PORT}.*${HOST}" 2>/dev/null || true

ssh -i "$KEY" -o IdentitiesOnly=yes -N -L "${PORT}:127.0.0.1:${PORT}" "${USER}@${HOST}" &
SSH_PID=$!
sleep 1
open "http://127.0.0.1:${PORT}"
echo "Live view: http://127.0.0.1:${PORT}  (tunnel pid $SSH_PID — stop with: kill $SSH_PID)"
wait "$SSH_PID"

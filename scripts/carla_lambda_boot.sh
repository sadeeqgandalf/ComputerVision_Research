#!/usr/bin/env bash
# ~/carla/boot.sh — max-quality CARLA on Lambda (GPU-only workload)
# Usage: cd ~/carla && ./boot.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
export DISPLAY=:99

echo "== stop old =="
# kill by name carefully (run from interactive shell)
kill $(pgrep -f 'CarlaUE4-Linux-Shipping' || true) 2>/dev/null || true
kill $(pgrep -f 'CarlaUE4.sh' || true) 2>/dev/null || true
kill $(pgrep -f 'Xvfb :99' || true) 2>/dev/null || true
sleep 2

echo "== Xvfb 1920x1080 =="
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1

echo "== CARLA Epic 1080p (A10) =="
cd "$ROOT"
# -vulkan if available often better on NVIDIA; fall back is fine
./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Epic \
  -ResX=1920 \
  -ResY=1080 \
  -benchmark \
  -fps=30 \
  >/tmp/carla.log 2>&1 &

echo "waiting for :2000 ..."
for _ in $(seq 1 90); do
  ss -ltn 2>/dev/null | grep -q ':2000 ' && break
  sleep 2
done
if ! ss -ltn 2>/dev/null | grep -q ':2000 '; then
  echo "CARLA failed — /tmp/carla.log:"
  tail -80 /tmp/carla.log
  exit 1
fi
echo "CARLA UP — Epic @ 1920x1080. Leave this terminal open (or it was backgrounded)."
echo "Next: python ~/carla/view_quality.py"
# keep shell attached to log if run interactively
tail -f /tmp/carla.log

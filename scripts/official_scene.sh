#!/usr/bin/env bash
# Official CARLA traffic (Traffic Manager) + optional browser stream.
# No custom driving logic — only CARLA examples + camera wrap.
#
# Prereq: CarlaUE4 already running (./boot.sh)
#
# Terminal 1 — built-in autopilot traffic:
#   bash ~/carla/official_scene.sh traffic
#
# Terminal 2 — stream only (watch in browser):
#   bash ~/carla/official_scene.sh stream
#
# Mac tunnel:
#   ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -N -L 8080:127.0.0.1:8080 ubuntu@170.9.12.5
#   open http://127.0.0.1:8080

set -euo pipefail
source "$HOME/venv-carla/bin/activate"
export PYTHONPATH="$HOME/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

cmd="${1:-}"
case "$cmd" in
  traffic)
    # Official example — set_autopilot + Traffic Manager (obeys lights/signs)
    cd "$HOME/carla/PythonAPI/examples"
    exec python generate_traffic.py -n 80 -w 20 --asynch --hero --safe
    ;;
  stream)
    # Camera MJPEG only — does not replace TM driving
    exec python "$HOME/carla/view_quality.py"
    ;;
  *)
    echo "usage: $0 {traffic|stream}"
    exit 1
    ;;
esac

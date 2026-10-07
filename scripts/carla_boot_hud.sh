#!/usr/bin/env bash
# Full CARLA HUD stack: map on UE cmdline, Xvfb + Vulkan (no RenderOffScreen).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" "$HOME/carla-runtime/logs" "$HOME/carla-runtime/recordings"
chmod 700 "$XDG_RUNTIME_DIR"

CARLA_LOG="$HOME/carla-runtime/logs/carla.log"
HUD_LOG="$HOME/carla-runtime/logs/hud.log"
VNC_LOG="$HOME/carla-runtime/logs/vnc.log"
XVFB_LOG="$HOME/carla-runtime/logs/xvfb.log"
NOVNC_LOG="$HOME/carla-runtime/logs/novnc.log"
TRAFFIC_LOG="$HOME/carla-runtime/logs/traffic.log"

DISPLAY_NUM="${DISPLAY_NUM:-99}"
export DISPLAY=":${DISPLAY_NUM}"
RESX="${RESX:-1920}"
RESY="${RESY:-1080}"
# UE map asset path stem (Content/Carla/Maps/<name>)
TOWN="${TOWN:-Town10HD_Opt}"
FPS="${FPS:-30}"
QUALITY="${QUALITY:-Epic}"
NVEH="${NVEH:-20}"
NWALK="${NWALK:-8}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
FILTER="${FILTER:-vehicle.tesla.model3}"
PREVIEW="${PREVIEW:-vnc}"   # vnc | none
HUD="${HUD:-manual}"        # manual | none  (none: server + display only, no hero, no traffic)

# NVIDIA Vulkan ICD (required for UE -vulkan on this host)
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export __GLX_VENDOR_LIBRARY_NAME=nvidia

EGG="$ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"
MAP_ARG="/Game/Carla/Maps/${TOWN}"

echo "== stop old stack =="
pkill -f 'CarlaUE4-Linux-Shipping' 2>/dev/null || true
pkill -f 'CarlaUE4.sh' 2>/dev/null || true
pkill -f 'manual_control.py' 2>/dev/null || true
pkill -f 'generate_traffic.py' 2>/dev/null || true
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "x11vnc.*:${VNC_PORT}" 2>/dev/null || true
pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
sleep 2

echo "== Xvfb ${RESX}x${RESY} (DISPLAY=${DISPLAY}) =="
Xvfb ":${DISPLAY_NUM}" -screen 0 "${RESX}x${RESY}x24" -ac +extension GLX +render -noreset \
  >"$XVFB_LOG" 2>&1 &
sleep 1

echo "== CARLA ${QUALITY} ${RESX}x${RESY} @ ${FPS}fps  map=${MAP_ARG}  RHI=Vulkan =="
echo "    VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
cd "$ROOT"
# Chrono + UE deps must be on LD_LIBRARY_PATH for Shipping binary.
CHRONO_LIB="$ROOT/CarlaUE4/Plugins/Carla/CarlaDependencies/lib"
export LD_LIBRARY_PATH="${CHRONO_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Map on cmdline — no Client.load_world() race.
# No -RenderOffScreen — UE renders into Xvfb via Vulkan.
nohup env \
  DISPLAY="$DISPLAY" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}" \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  ./CarlaUE4.sh "${MAP_ARG}" \
    -carla-rpc-port=2000 \
    -carla-streaming-port=2001 \
    -vulkan \
    -nosound \
    -quality-level="${QUALITY}" \
    -ResX="${RESX}" \
    -ResY="${RESY}" \
    -benchmark \
    -fps="${FPS}" \
  >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!

echo "waiting for RPC + GPU (pid $CARLA_PID) ..."
ok=0
for i in $(seq 1 90); do
  if ! kill -0 "$CARLA_PID" 2>/dev/null && ! pgrep -f CarlaUE4-Linux-Shipping >/dev/null; then
    echo "CARLA died during boot — tail:"
    tail -80 "$CARLA_LOG"
    exit 1
  fi
  rpc=0
  ss -ltn 2>/dev/null | grep -q ':2000 ' && rpc=1
  mem=0
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print int($1)}' | head -1 || echo 0)
  if [[ "$rpc" -eq 1 && "$mem" -gt 500 ]]; then
    ok=1
    echo "CARLA UP  rpc=1  gpu_mem=${mem}MiB  iter=$i"
    break
  fi
  sleep 2
done
if [[ "$ok" -ne 1 ]]; then
  echo "CARLA health gate failed (need :2000 + GPU mem>500) — tail:"
  tail -100 "$CARLA_LOG"
  nvidia-smi || true
  exit 1
fi

source "$HOME/venv-carla/bin/activate"
export PYTHONPATH="${EGG}:${PYTHONPATH:-}"

echo "== client ping + weather (map already loaded by UE) =="
python - <<PY
import carla
c = carla.Client("127.0.0.1", 2000)
c.set_timeout(60.0)
w = c.get_world()
print("map", w.get_map().name)
# Winter / snowy look (0.9.14 has no HardSnow preset — approximate with wet+fog+overcast)
wp = carla.WeatherParameters.WetCloudyNoon
wp.cloudiness = 100.0
wp.precipitation = 80.0
wp.precipitation_deposits = 100.0
wp.wetness = 100.0
wp.fog_density = 40.0
wp.fog_distance = 8.0
wp.wind_intensity = 60.0
wp.sun_altitude_angle = 8.0
w.set_weather(wp)
print("weather snowy/winter WetCloudyNoon+")
print("map", w.get_map().name)
PY

if [[ "$PREVIEW" == "vnc" ]]; then
  echo "== x11vnc + noVNC (preview only — RFB latency expected) =="
  x11vnc -display ":${DISPLAY_NUM}" -rfbport "${VNC_PORT}" -forever -shared -nopw -xkb -repeat \
    -speeds lan -threads \
    >"$VNC_LOG" 2>&1 &
  NOVNC_WEB=""
  for p in /usr/share/novnc /usr/share/novnc/utils/..; do
    [[ -d "$p" ]] && NOVNC_WEB="$p" && break
  done
  websockify --web="${NOVNC_WEB}" "${NOVNC_PORT}" "localhost:${VNC_PORT}" \
    >"$NOVNC_LOG" 2>&1 &
fi

if [[ "$HUD" == "manual" ]]; then
  echo "== pygame HUD (manual_control) ${RESX}x${RESY} =="
  cd "$ROOT/PythonAPI/examples"
  nohup env DISPLAY="$DISPLAY" SDL_VIDEODRIVER=x11 \
    python manual_control.py \
      --res "${RESX}x${RESY}" \
      --filter "${FILTER}" \
      -a \
    >"$HUD_LOG" 2>&1 &
  HUD_PID=$!
  sleep 6
  if ! kill -0 "$HUD_PID" 2>/dev/null; then
    echo "HUD failed — tail:"
    tail -80 "$HUD_LOG"
    exit 1
  fi

  echo "== traffic TM n=${NVEH} w=${NWALK} =="
  nohup python generate_traffic.py -n "${NVEH}" -w "${NWALK}" --asynch --safe \
    >"$TRAFFIC_LOG" 2>&1 &
else
  echo "== HUD=none: no manual_control, no traffic (the client spawns its own hero) =="
fi

# record helper next to boot script
cat > "$ROOT/record_mp4.sh" << 'REC'
#!/usr/bin/env bash
set -euo pipefail
SEC="${1:-30}"
OUT="$HOME/carla-runtime/recordings/carla_hud_$(date +%Y%m%d_%H%M%S).mp4"
mkdir -p "$(dirname "$OUT")"
ffmpeg -y -loglevel error -f x11grab -video_size "${RESX:-1920}x${RESY:-1080}" -framerate 30 -i "${DISPLAY:-:99}.0" \
  -c:v libx264 -preset veryfast -crf 17 -pix_fmt yuv420p -movflags +faststart -t "$SEC" "$OUT"
ls -lh "$OUT"
echo "$OUT"
REC
chmod +x "$ROOT/record_mp4.sh"

cat <<MSG

========================================
CARLA HUD STACK READY
  Boot   : map on UE cmdline (${MAP_ARG})
  RHI    : Vulkan  ICD=${VK_ICD_FILENAMES:-default}
  Render : Xvfb ${DISPLAY}  (no -RenderOffScreen)
  Server : ${QUALITY} ${RESX}x${RESY} @ ${FPS}fps
  HUD    : ${HUD}  filter=${FILTER}
  Traffic: $([[ "$HUD" == "manual" ]] && echo "${NVEH} veh / ${NWALK} walk" || echo "none (client spawns)")
  Preview: ${PREVIEW}  -> http://127.0.0.1:${NOVNC_PORT}/vnc.html
  Record : ${ROOT}/record_mp4.sh 30

  ssh -L ${NOVNC_PORT}:127.0.0.1:${NOVNC_PORT} ubuntu@HOST
  Keys: WASD | P autopilot | F1 HUD | C weather | H help
========================================
MSG

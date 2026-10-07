#!/usr/bin/env python3
"""Max-quality fullscreen RGB view for CARLA (run ON Lambda).

Epic-looking camera stream sized for a 16:9 monitor. Browser goes fullscreen.

  source ~/venv-carla/bin/activate
  export PYTHONPATH=~/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg:$PYTHONPATH
  python ~/carla/view_quality.py

Mac tunnel:
  ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -N -L 8080:127.0.0.1:8080 ubuntu@170.9.12.5
  open http://127.0.0.1:8080  → press F / click for fullscreen
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


INDEX_HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CARLA</title>
<style>
  html, body { margin:0; height:100%; background:#000; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:flex; align-items:center; justify-content:center; }
  img { width:100vw; height:100vh; object-fit:contain; background:#000; cursor:pointer; }
  #hint {
    position:fixed; left:12px; bottom:12px; color:rgba(255,255,255,.55);
    font:13px/1.4 system-ui,sans-serif; pointer-events:none;
  }
</style>
</head>
<body>
  <div id="wrap"><img id="v" src="/stream" alt="carla"/></div>
  <div id="hint">click video or press F = fullscreen</div>
  <script>
    const img = document.getElementById('v');
    function goFS() {
      const el = document.documentElement;
      if (!document.fullscreenElement) el.requestFullscreen?.();
      else document.exitFullscreen?.();
    }
    img.addEventListener('click', goFS);
    window.addEventListener('keydown', e => { if (e.key === 'f' || e.key === 'F') goFS(); });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=2000)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--town", default="", help="empty = keep current world")
    parser.add_argument("--fov", type=float, default=90.0)
    args = parser.parse_args()
    frame_interval = 1.0 / args.fps

    import carla  # type: ignore
    import numpy as np
    import cv2

    print(f"connecting {args.host}:{args.rpc_port} ...")
    client = carla.Client(args.host, args.rpc_port)
    client.set_timeout(60.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    bp = world.get_blueprint_library()

    # clear leftover heroes/sensors from prior runs
    for a in list(world.get_actors().filter("sensor.*")):
        a.destroy()

    vehicles = list(world.get_actors().filter("vehicle.*"))
    hero = None
    for v in vehicles:
        if v.attributes.get("role_name") == "hero":
            hero = v
            break
    if hero is None:
        hero_bp = bp.filter("vehicle.tesla.model3")[0]
        hero_bp.set_attribute("role_name", "hero")
        spawn = world.get_map().get_spawn_points()[0]
        hero = world.spawn_actor(hero_bp, spawn)
    # Built-in CARLA driving = Traffic Manager autopilot
    hero.set_autopilot(True)
    print("streaming hero id=%s (TM autopilot ON)" % hero.id)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", f"{frame_interval:.6f}")
    # richer look
    if cam_bp.has_attribute("enable_postprocess_effects"):
        cam_bp.set_attribute("enable_postprocess_effects", "true")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=-6.5, z=3.2), carla.Rotation(pitch=-12)),
        attach_to=hero,
    )

    latest = {"jpeg": None, "lock": threading.Lock()}
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    def on_image(image: "carla.Image") -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3]
        ok, buf = cv2.imencode(".jpg", arr, jpeg_params)
        if ok:
            with latest["lock"]:
                latest["jpeg"] = buf.tobytes()

    camera.listen(on_image)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(INDEX_HTML)))
                self.end_headers()
                self.wfile.write(INDEX_HTML)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with latest["lock"]:
                        frame = latest["jpeg"]
                    if frame is None:
                        time.sleep(frame_interval)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(frame_interval)
            except (BrokenPipeError, ConnectionResetError):
                return

    httpd = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
    print(
        f"QUALITY stream {args.width}x{args.height} @ {args.fps:g}fps q={args.jpeg_quality} — "
        f"http://127.0.0.1:{args.http_port}  (F = fullscreen)"
    )
    try:
        httpd.serve_forever()
    finally:
        camera.stop()
        camera.destroy()


if __name__ == "__main__":
    main()

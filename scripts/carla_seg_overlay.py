#!/usr/bin/env python3
"""RGB + CityScapes semantic masks for the live CARLA hero.

Modes:
  record (default)  write write MP4 then exit
  --live             pygame window on DISPLAY (for noVNC / Xvfb)
  --live --out PATH  live view AND record simultaneously

Does not restart the world. Optional --rain locks HardRain weather.
"""
from __future__ import annotations

import argparse
import os
import time
import weakref
from collections import deque

import carla
import cv2
import numpy as np
from carla import ColorConverter as cc


def _find_hero(world: carla.World):
    for v in world.get_actors().filter("vehicle.*"):
        if v.attributes.get("role_name", "") == "hero":
            return v
    for v in world.get_actors().filter("vehicle.tesla.*"):
        return v
    vehs = list(world.get_actors().filter("vehicle.*"))
    return vehs[0] if vehs else None


def _bgra_to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def lock_rain(world: carla.World) -> None:
    wp = world.get_weather()
    wp.cloudiness = 100.0
    wp.precipitation = 100.0
    wp.precipitation_deposits = 100.0
    wp.wetness = 100.0
    wp.wind_intensity = 50.0
    wp.fog_density = 10.0
    wp.sun_altitude_angle = 20.0
    world.set_weather(wp)


def ensure_autopilot(client: carla.Client, hero: carla.Actor, tm_port: int = 8000) -> None:
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(False)
    hero.set_autopilot(True, tm_port)
    try:
        tm.vehicle_percentage_speed_difference(hero, -15)
        tm.ignore_lights_percentage(hero, 20)
    except Exception:
        pass


class CameraPair:
    def __init__(self, world: carla.World, hero: carla.Actor, width: int, height: int, fov: float):
        self._rgb_q: deque = deque(maxlen=2)
        self._seg_q: deque = deque(maxlen=2)
        bp = world.get_blueprint_library()
        rgb_bp = bp.find("sensor.camera.rgb")
        seg_bp = bp.find("sensor.camera.semantic_segmentation")
        for cam in (rgb_bp, seg_bp):
            cam.set_attribute("image_size_x", str(width))
            cam.set_attribute("image_size_y", str(height))
            cam.set_attribute("fov", str(fov))

        transform = carla.Transform(
            carla.Location(x=-8.0, z=3.5),
            carla.Rotation(pitch=-12.0),
        )
        self.rgb = world.spawn_actor(rgb_bp, transform, attach_to=hero)
        self.seg = world.spawn_actor(seg_bp, transform, attach_to=hero)
        weak_self = weakref.ref(self)

        def _on_rgb(img, s=weak_self):
            self_ = s()
            if self_ is not None:
                self_._rgb_q.append(img)

        def _on_seg(img, s=weak_self):
            self_ = s()
            if self_ is not None:
                self_._seg_q.append(img)

        self.rgb.listen(_on_rgb)
        self.seg.listen(_on_seg)

    def pop_pair(self, timeout: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._rgb_q and self._seg_q:
                return self._rgb_q.pop(), self._seg_q.pop()
            time.sleep(0.01)
        return None, None

    def destroy(self):
        for s in (self.rgb, self.seg):
            if s is not None and s.is_alive:
                s.stop()
                s.destroy()


def compose(rgb_img: carla.Image, seg_img: carla.Image, mode: str, alpha: float):
    """Returns (display_bgr, mask_bgr)."""
    rgb = _bgra_to_bgr(rgb_img)
    seg = seg_img
    seg.convert(cc.CityScapesPalette)
    seg_bgr = _bgra_to_bgr(seg)

    if mode == "overlay":
        gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 8).astype(np.float32)[..., None]
        a = float(alpha) * mask
        out = (rgb.astype(np.float32) * (1.0 - a) + seg_bgr.astype(np.float32) * a).astype(np.uint8)
        return out, seg_bgr
    if mode == "side":
        return np.hstack([rgb, seg_bgr]), seg_bgr
    if mode == "seg":
        return seg_bgr, seg_bgr
    if mode == "triple":
        # RGB | overlay | pure mask — scaled to share width
        gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
        m = (gray > 8).astype(np.float32)[..., None]
        a = float(alpha) * m
        over = (rgb.astype(np.float32) * (1.0 - a) + seg_bgr.astype(np.float32) * a).astype(np.uint8)
        h, w = rgb.shape[:2]
        tw = w // 3
        panels = [cv2.resize(p, (tw, h), interpolation=cv2.INTER_AREA) for p in (rgb, over, seg_bgr)]
        return np.hstack(panels), seg_bgr
    raise ValueError(mode)


def draw_legend(frame: np.ndarray) -> np.ndarray:
    items = [
        ((128, 64, 128), "road"),
        ((244, 35, 232), "sidewalk"),
        ((70, 70, 70), "building"),
        ((102, 102, 156), "wall/fence"),
        ((220, 20, 60), "pedestrian"),
        ((255, 0, 0), "rider"),
        ((0, 0, 142), "car"),
        ((0, 0, 70), "truck"),
        ((0, 60, 100), "bus"),
        ((0, 0, 230), "motorcycle"),
        ((119, 11, 32), "bicycle"),
        ((70, 130, 180), "sky"),
        ((107, 142, 35), "vegetation"),
        ((152, 251, 152), "terrain"),
        ((255, 170, 30), "traffic light"),
        ((220, 220, 0), "traffic sign"),
        ((250, 170, 30), "pole"),
    ]
    out = frame.copy()
    x, y = 16, 16
    for bgr, name in items:
        cv2.rectangle(out, (x, y), (x + 18, y + 14), bgr, -1)
        cv2.putText(
            out,
            name,
            (x + 24, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
        if y > frame.shape[0] - 40:
            break
    return out


def draw_hud_banner(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, out.shape[0] - 36), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (16, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def run_live(args, client, world, hero, pair, writer, out_w, out_h):
    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.init()
    pygame.display.set_caption("CARLA semantic rain — overlay + mask")
    screen = pygame.display.set_mode((out_w, out_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    mode = args.mode
    alpha = args.alpha
    n = 0
    running = True
    t0 = time.time()
    t_end = t0 + args.seconds if args.seconds > 0 else None

    print("LIVE keys: 1=overlay 2=side 3=mask 4=triple  [ / ] alpha  Q/ESC quit", flush=True)
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_1:
                        mode = "overlay"
                    elif event.key == pygame.K_2:
                        mode = "side"
                    elif event.key == pygame.K_3:
                        mode = "seg"
                    elif event.key == pygame.K_4:
                        mode = "triple"
                    elif event.key == pygame.K_LEFTBRACKET:
                        alpha = max(0.1, alpha - 0.05)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        alpha = min(0.95, alpha + 0.05)

            if args.rain and n % 60 == 0:
                lock_rain(world)

            rgb_img, seg_img = pair.pop_pair(timeout=2.0)
            if rgb_img is None:
                clock.tick(args.fps)
                continue

            frame, _mask = compose(rgb_img, seg_img, mode, alpha)
            if args.legend and mode in ("overlay", "seg"):
                frame = draw_legend(frame)
            banner = "Town05 rain | mode=%s alpha=%.2f | 1 overlay 2 side 3 mask 4 triple | Q quit" % (
                mode,
                alpha,
            )
            frame = draw_hud_banner(frame, banner)
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if writer is not None:
                writer.write(frame)

            # pygame wants RGB
            surf = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            n += 1
            if n % args.fps == 0:
                print("live_frames", n, "mode", mode, flush=True)
            if t_end is not None and time.time() >= t_end:
                running = False
            clock.tick(args.fps)
    finally:
        pygame.quit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until quit (live)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--mode",
        choices=("overlay", "side", "seg", "triple"),
        default="overlay",
    )
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument(
        "--out",
        default="",
        help="optional MP4 path (record). Live can also write.",
    )
    ap.add_argument("--legend", action="store_true", default=True)
    ap.add_argument("--no-legend", action="store_false", dest="legend")
    ap.add_argument("--live", action="store_true", help="pygame window on $DISPLAY")
    ap.add_argument("--rain", action="store_true", help="force hard rain weather")
    ap.add_argument("--autopilot", action="store_true", default=True)
    ap.add_argument("--no-autopilot", action="store_false", dest="autopilot")
    args = ap.parse_args()

    if not args.live and not args.out:
        args.out = os.path.expanduser("~/carla-runtime/recordings/carla_seg_overlay.mp4")
    if not args.live and args.seconds <= 0:
        args.seconds = 30.0

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.rain:
        lock_rain(world)

    hero = _find_hero(world)
    if hero is None:
        raise SystemExit("no hero vehicle found")
    if args.autopilot:
        ensure_autopilot(client, hero)

    weather = world.get_weather()
    print(
        "map",
        world.get_map().name,
        "hero",
        hero.type_id,
        "precip",
        weather.precipitation,
        "veh",
        len(list(world.get_actors().filter("vehicle.*"))),
        "live",
        args.live,
        flush=True,
    )

    pair = CameraPair(world, hero, args.width, args.height, args.fov)
    time.sleep(1.0)

    if args.mode == "side":
        out_w, out_h = args.width * 2, args.height
    elif args.mode == "triple":
        # triple is composed at width already (== args.width)
        out_w, out_h = args.width, args.height
    else:
        out_w, out_h = args.width, args.height

    writer = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, float(args.fps), (out_w, out_h))
        if not writer.isOpened():
            pair.destroy()
            raise SystemExit("failed to open VideoWriter: %s" % args.out)

    n = 0
    try:
        if args.live:
            n = run_live(args, client, world, hero, pair, writer, out_w, out_h)
        else:
            t_end = time.time() + args.seconds
            while time.time() < t_end:
                rgb_img, seg_img = pair.pop_pair(timeout=2.0)
                if rgb_img is None:
                    print("warn: camera timeout", flush=True)
                    continue
                frame, _ = compose(rgb_img, seg_img, args.mode, args.alpha)
                if args.legend and args.mode in ("overlay", "seg"):
                    frame = draw_legend(frame)
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                n += 1
                if n % args.fps == 0:
                    print("frames", n, flush=True)
    finally:
        if writer is not None:
            writer.release()
        pair.destroy()

    if args.out:
        print("OUT=%s frames=%d" % (args.out, n), flush=True)
    else:
        print("DONE frames=%d" % n, flush=True)


if __name__ == "__main__":
    main()

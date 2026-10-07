#!/usr/bin/env python3
"""BlindBench BRAIN — CARLA runtime (not a scenario).

Owns: connect, sync world, ego@red, cameras, modal/amodal GT, panels, export, gates runner.
Scenarios live in scripts/scenario.py (copy/refine that file only).

Usage:
  python scripts/scenario.py --out /path/to/out
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def require_carla():
    try:
        import carla  # type: ignore
    except ImportError:
        print("Missing carla — set PYTHONPATH to egg + PythonAPI/carla.", file=sys.stderr)
        raise SystemExit(1)
    return carla


def dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def unit_dir(a, b) -> Tuple[float, float]:
    dx, dy = b.x - a.x, b.y - a.y
    n = math.sqrt(dx * dx + dy * dy) + 1e-9
    return dx / n, dy / n


def even_wh(w, h):
    return w - w % 2, h - h % 2


def build_K(w, h, fov_deg):
    import numpy as np

    f = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def as_mat4(m):
    import numpy as np

    if hasattr(m, "get_matrix"):
        m = m.get_matrix()
    return np.asarray(m, dtype=np.float64)


def project_point(loc, K, w2c):
    import numpy as np

    p = w2c @ np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float64)
    x, y, z = float(p[1]), float(-p[2]), float(p[0])
    if z <= 0.1:
        return None
    return K[0, 0] * (x / z) + K[0, 2], K[1, 1] * (y / z) + K[1, 2], z


def _box_from_uvs(us, vs, w, h, pad_px=12.0):
    if len(us) < 2:
        return None
    x0 = max(0.0, min(us) - pad_px)
    x1 = min(float(w - 1), max(us) + pad_px)
    y0 = max(0.0, min(vs) - pad_px)
    y1 = min(float(h - 1), max(vs) + pad_px)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def project_actor_bbox(actor, K, cam_tf, w, h, pad_px=8.0):
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    w2c = as_mat4(cam_tf.get_inverse_matrix())
    us, vs = [], []
    for v in verts:
        p = project_point(v, K, w2c)
        if p is None:
            continue
        us.append(p[0])
        vs.append(p[1])
    return _box_from_uvs(us, vs, w, h, pad_px=pad_px)


def project_walker_amodal(walker, K, cam_tf, w, h, pad_px=18.0):
    w2c = as_mat4(cam_tf.get_inverse_matrix())
    us, vs = [], []
    try:
        for bt in walker.get_bones().bone_transforms:
            p = project_point(bt.world.location, K, w2c)
            if p is None:
                continue
            us.append(p[0])
            vs.append(p[1])
    except Exception:
        pass
    bbox = project_actor_bbox(walker, K, cam_tf, w, h, pad_px=0.0)
    if bbox is not None:
        us.extend([bbox[0], bbox[2]])
        vs.extend([bbox[1], bbox[3]])
    return _box_from_uvs(us, vs, w, h, pad_px=pad_px)


def rgb_of(image):
    import numpy as np

    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    return a.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()


def inst_of(image):
    import numpy as np

    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgr = a.reshape((image.height, image.width, 4))[:, :, :3]
    return (bgr[:, :, 1].astype(np.int32) << 8) + bgr[:, :, 2].astype(np.int32)


def mask_iid(inst, iid):
    import numpy as np

    return ((inst == int(iid)).astype(np.uint8) * 255)


def lock_iid(inst, amodal, min_px=80):
    import numpy as np

    if amodal is None:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in amodal]
    h, w = inst.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = inst[y0:y1, x0:x1]
    ids, counts = np.unique(crop, return_counts=True)
    for i in np.argsort(-counts):
        iid = int(ids[i])
        if iid != 0 and counts[i] >= min_px:
            return iid
    return None


def occ_ratio(modal_area, ref):
    if ref <= 1:
        return 0.0
    return max(0.0, min(1.0, 1.0 - modal_area / ref))


def find_ego_at_red(world, carla):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    spawns = world.get_map().get_spawn_points()
    best, best_d, best_tl = None, 1e9, None
    for sp in spawns:
        for tl in lights:
            d = sp.location.distance(tl.get_location())
            if not (8.0 < d < 28.0):
                continue
            fwd = sp.get_forward_vector()
            to_tl = tl.get_location() - sp.location
            norm = math.sqrt(to_tl.x ** 2 + to_tl.y ** 2) + 1e-6
            align = (fwd.x * to_tl.x + fwd.y * to_tl.y) / norm
            if align > 0.35 and d < best_d:
                best, best_d, best_tl = sp, d, tl
    if best is None and spawns:
        best = spawns[0]
        if lights:
            best_tl = min(lights, key=lambda t: best.location.distance(t.get_location()))
    return best, best_tl


def freeze_red(tl, carla):
    if tl is None:
        return
    try:
        tl.set_state(carla.TrafficLightState.Red)
        tl.freeze(True)
        for other in tl.get_group_traffic_lights():
            other.set_state(carla.TrafficLightState.Red)
            other.freeze(True)
    except Exception:
        try:
            tl.set_state(carla.TrafficLightState.Red)
            tl.freeze(True)
        except Exception:
            pass


def draw_amodal(img, amodal, actor, K, cam_tf, color=(0, 255, 255), thickness=2):
    import cv2

    if amodal is not None:
        x0, y0, x1, y1 = map(int, amodal)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    verts = list(actor.bounding_box.get_world_vertices(actor.get_transform()))
    w2c = as_mat4(cam_tf.get_inverse_matrix())
    pts = []
    for v in verts:
        p = project_point(v, K, w2c)
        pts.append(None if p is None else (int(round(p[0])), int(round(p[1]))))
    H, W = img.shape[:2]
    for a, b in _BOX_EDGES:
        if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
            continue
        pa, pb = pts[a], pts[b]
        if max(abs(pa[0]), abs(pb[0])) > 4 * W or max(abs(pa[1]), abs(pb[1])) > 4 * H:
            continue
        cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)


def make_panel(rgb, modal, ped, K, cam_tf, amodal, occ, frame_i, n, phase, footer: str):
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    modal_area = int((modal > 0).sum())
    left = rgb.copy()
    if modal_area > 0:
        tint = np.zeros_like(left)
        tint[:, :, 1] = modal
        left = cv2.addWeighted(left, 1.0, tint, 0.4, 0)
    draw_amodal(left, amodal, ped, K, cam_tf, (0, 255, 255), 2)
    bw = int(w * 0.32)
    x0, y0 = 20, 18
    cv2.rectangle(left, (x0, y0), (x0 + bw, y0 + 20), (25, 25, 25), -1)
    fill = int(bw * occ)
    col = (0, 170, 0) if occ < 0.35 else ((0, 165, 255) if occ < 0.85 else (0, 0, 220))
    cv2.rectangle(left, (x0, y0), (x0 + fill, y0 + 20), col, -1)
    pct = 100.0 * modal_area / max(1.0, float(w * h))
    cv2.putText(
        left,
        f"occ={occ:.2f}  visible={modal_area}px ({pct:.2f}%)  {frame_i}/{n}",
        (x0, y0 + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(left, phase, (x0, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(left, footer, (x0, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 2, cv2.LINE_AA)

    mid = np.zeros_like(rgb)
    mid[:, :, 1] = modal
    mid[modal == 0] = (18, 18, 18)
    cv2.putText(mid, "MODAL = visible pixels only", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(mid, f"{modal_area} px", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)

    right = (rgb.astype(np.float32) * (0.22 if occ >= 0.85 else 0.55)).astype(np.uint8)
    draw_amodal(right, amodal, ped, K, cam_tf, (0, 255, 255), 3)
    cv2.putText(right, "AMODAL = bones+bbox", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if occ >= 0.95:
        cv2.putText(right, "HIDDEN — amodal must persist", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    gap = np.full((h, 8, 3), 35, dtype=np.uint8)
    return np.concatenate([left, gap, mid, gap, right], axis=1)


@dataclass
class SceneContext:
    """Passed into scenario.setup / scenario.on_tick."""

    carla: Any
    world: Any
    bp: Any
    ego: Any
    tl: Any
    et: Any
    fwd: Any
    right: Any
    origin: Any
    road_z: float
    dt: float
    width: int
    height: int
    fov: float
    seed: int = 11

    def loc(self, depth: float, lat: float, z: Optional[float] = None):
        zz = (self.road_z + 0.5) if z is None else z
        return self.carla.Location(
            self.origin.x + self.fwd.x * depth + self.right.x * lat,
            self.origin.y + self.fwd.y * depth + self.right.y * lat,
            zz,
        )


@dataclass
class SceneActors:
    """What a scenario must return from setup()."""

    ped: Any
    occluder: Any
    tick: Callable[[int], str]  # frame_i -> phase string
    footer: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    # optional: called each grab before sensors (keep parked brake, etc.)
    hold: Optional[Callable[[], None]] = None
    # optional: called once after ped-id lock, before recording loop
    on_record_start: Optional[Callable[[], None]] = None


class BlindBenchBrain:
    """CARLA capture runtime. Scenario plugins provide spawn + per-tick motion."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        width: int = 1280,
        height: int = 720,
        fov: float = 70.0,
        dt: float = 0.05,
        frames: int = 280,
        town: str = "",
        seed: int = 11,
    ):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fov = fov
        self.dt = dt
        self.frames = frames
        self.town = town
        self.seed = seed
        random.seed(seed)

    def run(self, scenario, out: Path) -> Dict[str, Any]:
        import cv2
        import numpy as np

        carla = require_carla()
        out = Path(out)
        for sub in ("rgb", "modal", "panels"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        client = carla.Client(self.host, self.port)
        client.set_timeout(90.0)
        world = client.load_world(self.town) if self.town else client.get_world()
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)

        for a in list(world.get_actors()):
            try:
                if a.type_id.startswith(("sensor.", "vehicle.", "walker.", "controller.")):
                    a.destroy()
            except Exception:
                pass
        world.tick()

        bp = world.get_blueprint_library()
        ego_sp, tl = find_ego_at_red(world, carla)
        freeze_red(tl, carla)

        ego_bp = bp.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "blindbench_ego")
        ego = None
        for sp in [ego_sp] + list(world.get_map().get_spawn_points()):
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego:
                break
        if ego is None:
            raise RuntimeError("ego spawn failed")
        world.tick()

        et = ego.get_transform()
        ctx = SceneContext(
            carla=carla,
            world=world,
            bp=bp,
            ego=ego,
            tl=tl,
            et=et,
            fwd=et.get_forward_vector(),
            right=et.get_right_vector(),
            origin=et.location,
            road_z=world.get_map().get_waypoint(et.location).transform.location.z,
            dt=self.dt,
            width=self.width,
            height=self.height,
            fov=self.fov,
            seed=self.seed,
        )

        actors = scenario.setup(ctx)
        ped, occ = actors.ped, actors.occluder

        ego.set_autopilot(False)
        brake = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        ego.apply_control(brake)
        freeze_red(tl, carla)
        world.tick()

        cam_bp = bp.find("sensor.camera.rgb")
        inst_bp = bp.find("sensor.camera.instance_segmentation")
        for s in (cam_bp, inst_bp):
            s.set_attribute("image_size_x", str(self.width))
            s.set_attribute("image_size_y", str(self.height))
            s.set_attribute("fov", str(self.fov))
        cam_tf = carla.Transform(carla.Location(x=1.4, z=1.5), carla.Rotation(pitch=-4.0))
        rgb_cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        inst_cam = world.spawn_actor(inst_bp, cam_tf, attach_to=ego)
        latest: Dict[str, Any] = {"rgb": None, "inst": None}
        rgb_cam.listen(lambda im: latest.__setitem__("rgb", im))
        inst_cam.listen(lambda im: latest.__setitem__("inst", im))

        def grab():
            latest["rgb"] = latest["inst"] = None
            ego.apply_control(brake)
            if actors.hold:
                actors.hold()
            freeze_red(tl, carla)
            world.tick()
            for _ in range(100):
                if latest["rgb"] is not None and latest["inst"] is not None:
                    return latest["rgb"], latest["inst"]
                time.sleep(0.001)
            return latest["rgb"], latest["inst"]

        # warmup
        for _ in range(25):
            actors.tick(0)
            grab()

        K = build_K(self.width, self.height, self.fov)
        locked = None
        ref_modal = 0.0
        for _ in range(80):
            actors.tick(0)
            rgb_img, inst_img = grab()
            if rgb_img is None or inst_img is None:
                continue
            inst = inst_of(inst_img)
            amodal = project_walker_amodal(ped, K, rgb_cam.get_transform(), self.width, self.height)
            iid = lock_iid(inst, amodal, min_px=40)
            if iid is None:
                continue
            area = int((mask_iid(inst, iid) > 0).sum())
            if 150 <= area < 0.02 * self.width * self.height:
                locked, ref_modal = iid, float(max(ref_modal, area))
                if ref_modal >= 300:
                    break
        if locked is None or ref_modal < 150:
            raise RuntimeError(f"ped lock failed (iid={locked}, ref={ref_modal})")

        print(
            f"LOCKED ped={ped.id} iid={locked} ref={int(ref_modal)}px "
            f"({100 * ref_modal / (self.width * self.height):.2f}%) scenario={scenario.name}",
            flush=True,
        )

        if actors.on_record_start:
            actors.on_record_start()
            # Re-acquire a ped-sized instance after scenario resets pose.
            world.tick()
            rgb_img, inst_img = grab()
            if rgb_img is not None and inst_img is not None:
                inst = inst_of(inst_img)
                max_ped = 0.015 * self.width * self.height
                area0 = int((mask_iid(inst, locked) > 0).sum())
                if 80 <= area0 < max_ped:
                    ref_modal = float(max(ref_modal, area0))
                else:
                    amodal = project_walker_amodal(
                        ped, K, rgb_cam.get_transform(), self.width, self.height
                    )
                    iid = lock_iid(inst, amodal, min_px=40)
                    if iid is not None:
                        area = int((mask_iid(inst, iid) > 0).sum())
                        if 80 <= area < max_ped:
                            locked, ref_modal = iid, float(area)
                            print(
                                f"RELOCKED ped iid={locked} ref={int(ref_modal)}px",
                                flush=True,
                            )

        n = self.frames
        pw, ph = even_wh(self.width * 3 + 16, self.height)
        buffered, records = [], []
        ped_steps, occ_steps = [], []
        prev_ped, prev_occ = ped.get_location(), occ.get_location()

        for i in range(n):
            phase = actors.tick(i)
            rgb_img, inst_img = grab()
            if rgb_img is None or inst_img is None:
                continue

            ped_loc, occ_loc = ped.get_location(), occ.get_location()
            ped_steps.append(dist3(prev_ped, ped_loc))
            occ_steps.append(dist3(prev_occ, occ_loc))
            prev_ped, prev_occ = ped_loc, occ_loc

            rgb = rgb_of(rgb_img)
            inst = inst_of(inst_img)
            modal = mask_iid(inst, locked)
            cam_now = rgb_cam.get_transform()
            amodal = project_walker_amodal(ped, K, cam_now, self.width, self.height)
            modal_area = int((modal > 0).sum())
            if modal_area > ref_modal and "OCCLUSION" not in phase.upper():
                ref_modal = float(modal_area)
            occ_v = occ_ratio(modal_area, ref_modal)
            full = bool(modal_area <= max(15, 0.08 * ref_modal) and amodal is not None and ref_modal >= 150)

            cover_ok = True
            if amodal is not None and modal_area > 0:
                ys, xs = np.where(modal > 0)
                if len(xs):
                    cover_ok = bool(
                        xs.min() >= amodal[0] - 2
                        and xs.max() <= amodal[2] + 2
                        and ys.min() >= amodal[1] - 2
                        and ys.max() <= amodal[3] + 2
                    )

            panel = make_panel(
                rgb, modal, ped, K, cam_now, amodal, occ_v, i, n - 1, phase, actors.footer
            )
            if panel.shape[1] != pw or panel.shape[0] != ph:
                panel = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
            buffered.append((i, rgb, modal, panel))
            rec = {
                "frame": i,
                "phase": phase,
                "modal_area_px": modal_area,
                "modal_pct_of_frame": 100.0 * modal_area / float(self.width * self.height),
                "ref_modal_area_px": ref_modal,
                "amodal_bbox_xyxy": amodal,
                "occlusion_ratio": occ_v,
                "fully_occluded": full,
                "amodal_covers_modal": cover_ok,
                "ped_step_m": ped_steps[-1],
                "occ_step_m": occ_steps[-1],
            }
            records.append(rec)
            if i % 30 == 0 or full:
                print(
                    f"[{i}/{n}] {phase[:40]} | modal={modal_area} occ={occ_v:.2f} "
                    f"pedΔ={ped_steps[-1]:.3f} occΔ={occ_steps[-1]:.3f}",
                    flush=True,
                )

        dt = self.dt
        ped_speeds = [s / dt for s in ped_steps[1:]]
        occ_speeds = [s / dt for s in occ_steps[1:]]
        motion = {
            "ped_max_step_m": max(ped_steps[5:], default=0),
            "occ_max_step_m": max(occ_steps[5:], default=0),
            "ped_max_speed_mps": max(ped_speeds[5:], default=0),
            "occ_max_speed_mps": max(occ_speeds[5:], default=0),
            "ped_mean_speed_mps": float(np.mean(ped_speeds[5:])) if ped_speeds[5:] else 0,
            "ped_teleport": max(ped_steps[5:], default=0) > 1.2,
            "occ_teleport": max(occ_steps[5:], default=0) > 0.5,
            "ped_not_moving": float(np.mean(ped_speeds[10:60])) < 0.2 if len(ped_speeds) > 60 else True,
        }

        quality = scenario.evaluate(records, motion)
        quality_pass = bool(quality.get("pass", False))

        summary = {
            "scenario": scenario.name,
            "narrative": getattr(scenario, "narrative", ""),
            "frames": len(records),
            "motion_audit": motion,
            "quality_gates": quality,
            "quality_pass": quality_pass,
            "meta": actors.meta,
            "map": world.get_map().name,
            "out": str(out),
            "modal_explained": {
                "modal_area_px": "visible ped pixels",
                "amodal": "bones + bbox padded",
                "occlusion_ratio": "1 - modal_now/modal_ref in [0,1]",
            },
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"quality_pass": quality_pass, "quality_gates": quality}, indent=2), flush=True)

        gt_path = out / "gt.jsonl"
        if not quality_pass:
            print("REJECTED — debug only.", file=sys.stderr)
            with gt_path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            dbg = cv2.VideoWriter(
                str(out / "debug_reject.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (pw, ph)
            )
            for idx, (_, _, _, panel) in enumerate(buffered):
                if idx % 6 == 0:
                    dbg.write(panel[:, :, ::-1])
            dbg.release()
        else:
            writer = cv2.VideoWriter(
                str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / self.dt, (pw, ph)
            )
            with gt_path.open("w") as f:
                for (i, rgb, modal, panel), rec in zip(buffered, records):
                    stem = f"{i:06d}"
                    cv2.imwrite(str(out / "rgb" / f"{stem}.jpg"), rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    cv2.imwrite(str(out / "modal" / f"{stem}.png"), modal)
                    cv2.imwrite(str(out / "panels" / f"{stem}.jpg"), panel[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    writer.write(panel[:, :, ::-1])
                    f.write(json.dumps(rec) + "\n")
            writer.release()
            print("FINAL OK", flush=True)

        rgb_cam.stop()
        inst_cam.stop()
        for a in (rgb_cam, inst_cam, ego, ped, occ):
            try:
                a.destroy()
            except Exception:
                pass
        if tl is not None:
            try:
                tl.freeze(False)
            except Exception:
                pass
        world.apply_settings(original)
        if not quality_pass:
            raise SystemExit(2)
        return summary

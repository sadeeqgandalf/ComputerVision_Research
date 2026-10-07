#!/usr/bin/env python3
"""
BlindBench — Research Occlusion Scenario
=========================================

Purpose
-------
Generate a physically plausible pedestrian occlusion event for CARLA.

Research phenomenon
-------------------
    VISIBLE
        ↓
    PARTIAL OCCLUSION
        ↓
    SEVERE OCCLUSION
        ↓
    FULL / NEAR-FULL OCCLUSION
        ↓
    EMERGENCE
        ↓
    VISIBLE

The scenario creates the geometry.
BlindBenchBrain measures modal/amodal visibility.

IMPORTANT
---------
This file intentionally does NOT modify BlindBenchBrain.

The Brain already provides:
    - RGB camera
    - instance segmentation
    - persistent pedestrian instance ID
    - modal visible pixels
    - amodal pedestrian projection
    - occlusion ratio
    - full-occlusion detection
    - motion audit

The scenario therefore focuses on:
    - physically plausible actor placement
    - deterministic pedestrian trajectory
    - stationary occluder
    - camera-view occlusion geometry
    - explicit experimental phases
    - research metadata
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blindbench_brain import BlindBenchBrain, SceneActors, dist3, unit_dir


# =============================================================================
# Shared camera-space helpers (phase HINTS only — Brain measures GT)
# =============================================================================

def _camera_projection(ctx, world_point):
    """Coarse ego-camera projection for scenario phase labels only."""
    cam_tf = ctx.carla.Transform(
        ctx.carla.Location(x=1.4, z=1.5),
        ctx.carla.Rotation(pitch=-4.0),
    )
    ego_tf = ctx.et
    cam_world = ego_tf.transform(cam_tf.location)
    cam_rotation = ego_tf.rotation
    cam_world_rot = ctx.carla.Rotation(
        pitch=cam_rotation.pitch + cam_tf.rotation.pitch,
        yaw=cam_rotation.yaw + cam_tf.rotation.yaw,
        roll=cam_rotation.roll + cam_tf.rotation.roll,
    )
    rel = world_point - cam_world
    forward = cam_world_rot.get_forward_vector()
    right = cam_world_rot.get_right_vector()
    up = cam_world_rot.get_up_vector()
    z = rel.x * forward.x + rel.y * forward.y + rel.z * forward.z
    if z <= 0.1:
        return None
    x = rel.x * right.x + rel.y * right.y + rel.z * right.z
    y = rel.x * up.x + rel.y * up.y + rel.z * up.z
    f = ctx.width / (2.0 * math.tan(math.radians(ctx.fov) / 2.0))
    u = f * x / z + ctx.width / 2.0
    v = -f * y / z + ctx.height / 2.0
    return u, v, z


def _project_bbox(ctx, actor):
    points = []
    try:
        verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    except Exception:
        return None
    for vertex in verts:
        p = _camera_projection(ctx, vertex)
        if p is None:
            continue
        u, v, z = p
        if u < -ctx.width or u > 2 * ctx.width or v < -ctx.height or v > 2 * ctx.height:
            continue
        points.append((u, v, z))
    if len(points) < 2:
        return None
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    zs = [p[2] for p in points]
    return min(us), min(vs), max(us), max(vs), min(zs), max(zs)


def _bbox_overlap(a, b):
    """Intersection-over-area relative to bbox A (ped)."""
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area_a


def _pick_vehicle_bp(bp, preferred):
    for pattern in preferred:
        matches = bp.filter(pattern)
        if matches:
            return matches[0]
    vehicles = list(bp.filter("vehicle.*"))
    if not vehicles:
        raise RuntimeError("No vehicle blueprints available")
    return vehicles[0]


def _park_vehicle(carla, actor):
    actor.set_autopilot(False)
    actor.set_simulate_physics(True)
    ctrl = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    actor.apply_control(ctrl)
    return ctrl


class ParkedVanPedestrianOcclusion:
    """
    Controlled pedestrian-behind-parked-vehicle occlusion experiment.

    Coordinate convention from SceneContext:
        +depth = forward from ego
        +lat   = right of ego

    Experimental geometry:

                 ego camera
                     |
                     |
                     v

             parked vehicle
                 [ VAN ]
                    |
                    |     pedestrian trajectory
                    |          ---------->
                    |        /
                    |      /
                    |    /
                    |  /
                    |/
                pedestrian

    The pedestrian travels laterally across the scene at a greater
    depth than the parked vehicle, causing the vehicle to become
    the visual occluder from the ego camera viewpoint.
    """

    name = "parked_van_pedestrian_occlusion"

    narrative = (
        "Ego stopped at red. A parked van is staged FARTHER down the "
        "road (Experiment B vs 10m baseline). Pedestrian crosses behind "
        "it: visible → occluded → emerges. Controlled occluder-distance "
        "sweep."
    )

    def __init__(
        self,
        ped_speed: float = 1.35,
        van_depth: float = 14.0,
        van_lat: float = 4.0,
        ped_depth_offset: float = 3.0,
        ped_start_lat: float = -7.0,
        ped_end_lat: float = 8.0,
    ):
        self.ped_speed = float(ped_speed)

        # Geometry parameters.
        self.van_depth = float(van_depth)
        self.van_lat = float(van_lat)
        self.ped_depth_offset = float(ped_depth_offset)
        self.ped_start_lat = float(ped_start_lat)
        self.ped_end_lat = float(ped_end_lat)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _camera_projection(self, ctx, world_point):
        """
        Project a CARLA world point into the same camera coordinate
        convention used by BlindBenchBrain.

        This is used ONLY for coarse scenario phase estimation.

        Actual occlusion measurement remains the Brain's modal/amodal
        measurement system.
        """
        # Brain camera transform:
        # Location(x=1.4, z=1.5), pitch=-4 degrees.
        cam_tf = ctx.carla.Transform(
            ctx.carla.Location(x=1.4, z=1.5),
            ctx.carla.Rotation(pitch=-4.0),
        )

        # Build the camera transform relative to ego.
        ego_tf = ctx.et

        # Reconstruct the camera transform in world space.
        cam_world = ego_tf.transform(cam_tf.location)

        cam_rotation = ego_tf.rotation
        cam_world_rot = ctx.carla.Rotation(
            pitch=cam_rotation.pitch + cam_tf.rotation.pitch,
            yaw=cam_rotation.yaw + cam_tf.rotation.yaw,
            roll=cam_rotation.roll + cam_tf.rotation.roll,
        )

        # Camera-space approximation using ego basis.
        rel = world_point - cam_world

        forward = cam_world_rot.get_forward_vector()
        right = cam_world_rot.get_right_vector()
        up = cam_world_rot.get_up_vector()

        z = (
            rel.x * forward.x
            + rel.y * forward.y
            + rel.z * forward.z
        )

        if z <= 0.1:
            return None

        x = (
            rel.x * right.x
            + rel.y * right.y
            + rel.z * right.z
        )

        y = (
            rel.x * up.x
            + rel.y * up.y
            + rel.z * up.z
        )

        f = ctx.width / (
            2.0 * math.tan(math.radians(ctx.fov) / 2.0)
        )

        u = f * x / z + ctx.width / 2.0
        v = -f * y / z + ctx.height / 2.0

        return u, v, z

    def _project_bbox(self, ctx, actor):
        """
        Coarse image-space bounding box used only for phase labeling.

        The Brain remains authoritative for measured modal/amodal
        visibility.
        """
        points = []

        try:
            verts = actor.bounding_box.get_world_vertices(
                actor.get_transform()
            )
        except Exception:
            return None

        for vertex in verts:
            p = self._camera_projection(ctx, vertex)
            if p is None:
                continue

            u, v, z = p

            # Ignore wildly off-screen projections.
            if (
                u < -ctx.width
                or u > 2 * ctx.width
                or v < -ctx.height
                or v > 2 * ctx.height
            ):
                continue

            points.append((u, v, z))

        if len(points) < 2:
            return None

        us = [p[0] for p in points]
        vs = [p[1] for p in points]
        zs = [p[2] for p in points]

        return (
            min(us),
            min(vs),
            max(us),
            max(vs),
            min(zs),
            max(zs),
        )

    @staticmethod
    def _bbox_overlap(a, b):
        """
        Return intersection-over-area relative to bbox A.

        A = pedestrian bbox
        B = occluder bbox
        """
        ax0, ay0, ax1, ay1 = a[:4]
        bx0, by0, bx1, by1 = b[:4]

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)

        inter = iw * ih
        area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))

        return inter / area_a

    # ------------------------------------------------------------------
    # CARLA setup
    # ------------------------------------------------------------------

    def setup(self, ctx) -> SceneActors:
        carla = ctx.carla

        # Scenario-local deterministic randomness.
        random.seed(getattr(ctx, "seed", 11))

        # ==============================================================
        # PEDESTRIAN BLUEPRINT
        # ==============================================================

        ped_candidates = list(
            ctx.bp.filter("walker.pedestrian.*")
        )

        if not ped_candidates:
            raise RuntimeError("No pedestrian blueprints available")

        ped_bp = random.choice(ped_candidates)

        if ped_bp.has_attribute("is_invincible"):
            ped_bp.set_attribute("is_invincible", "true")

        # ==============================================================
        # OCCLUDER BLUEPRINT
        # ==============================================================

        # Prefer relatively large road vehicles so that the occlusion
        # comes from a plausible physical blocker rather than an
        # arbitrary tiny object.
        # Experiment B: same VW T2 family, closer staging so the
        # occluder dominates the FOV vs baseline @ 10m/3m.
        preferred = (
            "vehicle.volkswagen.t2",
            "vehicle.mercedes.sprinter",
            "vehicle.ford.ambulance",
            "vehicle.carlamotors.carlacola",
            "vehicle.toyota.prius",
            "vehicle.audi.a2",
        )

        occ_bp = None

        for pattern in preferred:
            matches = ctx.bp.filter(pattern)
            if matches:
                occ_bp = matches[0]
                break

        if occ_bp is None:
            vehicles = list(ctx.bp.filter("vehicle.*"))

            if not vehicles:
                raise RuntimeError("No vehicle blueprints available")

            occ_bp = vehicles[0]

        occ_bp.set_attribute(
            "role_name",
            "blindbench_static_occluder",
        )

        # ==============================================================
        # PARKED VEHICLE
        # ==============================================================

        occ = None

        # Small deterministic fallback set.
        candidate_positions = [
            (self.van_depth, self.van_lat),
            (self.van_depth + 0.5, self.van_lat),
            (self.van_depth - 0.5, self.van_lat),
            (self.van_depth, self.van_lat + 0.5),
            (self.van_depth, self.van_lat - 0.5),
            (self.van_depth + 1.0, self.van_lat - 1.0),
            (self.van_depth, self.van_lat - 1.0),
            (self.van_depth - 1.0, self.van_lat),
        ]

        actual_van_depth = None
        actual_van_lat = None

        for dep, lat in candidate_positions:
            transform = carla.Transform(
                ctx.loc(
                    dep,
                    lat,
                    ctx.road_z + 0.45,
                ),
                carla.Rotation(
                    yaw=ctx.et.rotation.yaw
                ),
            )

            candidate = ctx.world.try_spawn_actor(
                occ_bp,
                transform,
            )

            if candidate is not None:
                occ = candidate
                actual_van_depth = dep
                actual_van_lat = lat
                break

        if occ is None:
            raise RuntimeError(
                "Failed to spawn parked occluder"
            )

        occ.set_autopilot(False)
        occ.set_simulate_physics(True)

        parked_control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
        )

        occ.apply_control(parked_control)

        # ==============================================================
        # PEDESTRIAN TRAJECTORY
        # ==============================================================

        # Critical design choice:
        #
        # The pedestrian is placed farther forward than the van.
        #
        # From the ego camera:
        #
        #     camera → VAN → PEDESTRIAN
        #
        # This creates the possibility of genuine visual occlusion.
        ped_depth = (
            actual_van_depth + self.ped_depth_offset
        )

        ped_start = ctx.loc(
            ped_depth,
            self.ped_start_lat,
            ctx.road_z + 1.0,
        )

        ped_end = ctx.loc(
            ped_depth,
            self.ped_end_lat,
            ctx.road_z + 1.0,
        )

        direction_x, direction_y = unit_dir(
            ped_start,
            ped_end,
        )

        ped = None

        # Deterministic spawn fallbacks.
        for lat0 in (
            self.ped_start_lat,
            self.ped_start_lat - 0.5,
            self.ped_start_lat + 0.5,
            self.ped_start_lat - 1.0,
        ):
            spawn_location = ctx.loc(
                ped_depth,
                lat0,
                ctx.road_z + 1.0,
            )

            ped = ctx.world.try_spawn_actor(
                ped_bp,
                carla.Transform(
                    spawn_location,
                    carla.Rotation(
                        yaw=ctx.et.rotation.yaw + 90.0
                    ),
                ),
            )

            if ped is not None:
                ped_start = spawn_location
                break

        if ped is None:
            raise RuntimeError(
                "Failed to spawn pedestrian"
            )

        ctx.world.tick()

        # ==============================================================
        # STATE
        # ==============================================================

        state = {
            "last_phase": "VISIBLE_APPROACH",
            "entered_occlusion": False,
            "severe_occlusion_seen": False,
            "full_target_seen": False,
            "emerged": False,
        }

        # ==============================================================
        # PARKING HOLD
        # ==============================================================

        def hold():
            """
            Keep the occluder stationary.

            Brain calls this before each synchronous world tick.
            """
            occ.apply_control(parked_control)

        # ==============================================================
        # PEDESTRIAN MOTION
        # ==============================================================

        def tick_motion():
            here = ped.get_location()

            remaining = dist3(
                here,
                ped_end,
            )

            if remaining < 0.35:
                try:
                    ped.disable_constant_velocity()
                except Exception:
                    pass

                stop = carla.WalkerControl()
                stop.speed = 0.0
                stop.direction = carla.Vector3D(
                    direction_x,
                    direction_y,
                    0.0,
                )

                ped.apply_control(stop)
                return

            control = carla.WalkerControl()
            control.direction = carla.Vector3D(
                direction_x,
                direction_y,
                0.0,
            )
            control.speed = self.ped_speed
            control.jump = False
            ped.apply_control(control)

        # ==============================================================
        # PHASE ESTIMATION
        # ==============================================================

        def estimate_phase():
            """
            Estimate the intended geometric phase.

            IMPORTANT:
            This is NOT the authoritative occlusion detector.

            The Brain's modal/amodal measurements determine actual
            visibility. This function only provides useful HUD labels
            before the sensor measurements are available.
            """

            ped_box = self._project_bbox(
                ctx,
                ped,
            )

            occ_box = self._project_bbox(
                ctx,
                occ,
            )

            if ped_box is None or occ_box is None:
                return "VISIBLE_APPROACH"

            overlap = self._bbox_overlap(
                ped_box,
                occ_box,
            )

            ped_depth_cam = (
                ped_box[4] + ped_box[5]
            ) / 2.0

            occ_depth_cam = (
                occ_box[4] + occ_box[5]
            ) / 2.0

            # The blocker must actually be closer to the camera.
            front_to_ped = occ_depth_cam < ped_depth_cam

            if not front_to_ped or overlap < 0.03:
                if state["emerged"]:
                    return "VISIBLE_AFTER_OCCLUSION"

                return "VISIBLE_APPROACH"

            state["entered_occlusion"] = True

            if overlap >= 0.75:
                state["full_target_seen"] = True
                return "FULL_OCCLUSION_TARGET"

            if overlap >= 0.45:
                state["severe_occlusion_seen"] = True
                return "SEVERE_OCCLUSION"

            return "PARTIAL_OCCLUSION"

        def tick(frame_i: int) -> str:
            tick_motion()

            phase = estimate_phase()

            # Emergence requires projected overlap to clear AND the
            # pedestrian to remain inside a reasonable image margin
            # (walking off-FOV must not count as "emerged visible").
            ped_box = self._project_bbox(ctx, ped)
            occ_box = self._project_bbox(ctx, occ)

            if (
                state["entered_occlusion"]
                and ped_box is not None
                and occ_box is not None
            ):
                overlap = self._bbox_overlap(ped_box, occ_box)
                ped_depth_cam = (ped_box[4] + ped_box[5]) / 2.0
                occ_depth_cam = (occ_box[4] + occ_box[5]) / 2.0
                u_mid = 0.5 * (ped_box[0] + ped_box[2])
                in_frame = 40.0 < u_mid < (ctx.width - 40.0)

                if (
                    in_frame
                    and (overlap < 0.05 or occ_depth_cam >= ped_depth_cam)
                ):
                    here = ped.get_location()
                    rel = here - ctx.origin
                    lateral = (
                        rel.x * ctx.right.x
                        + rel.y * ctx.right.y
                    )
                    if lateral > actual_van_lat + 0.8:
                        state["emerged"] = True
                        phase = "VISIBLE_AFTER_OCCLUSION"

            state["last_phase"] = phase

            return phase

        # ==============================================================
        # METADATA
        # ==============================================================

        metadata = {
            "experiment": "pedestrian_occlusion_B_farther_van",
            "baseline_contrast": (
                "vs parked_003 / B0 (T2 @ 10.0m / 3.0m): "
                "T2 @ 14.0m / 4.0m — occluder distance sweep"
            ),
            "occlusion_mechanism": "parked_vehicle",
            "occluder_type": occ.type_id,
            "van_depth_m": actual_van_depth,
            "van_lateral_m": actual_van_lat,
            "pedestrian_depth_m": ped_depth,
            "pedestrian_depth_offset_m": self.ped_depth_offset,
            "pedestrian_start_lateral_m": self.ped_start_lat,
            "pedestrian_end_lateral_m": self.ped_end_lat,
            "pedestrian_speed_mps": self.ped_speed,
            "camera_fov_deg": ctx.fov,
            "camera_width_px": ctx.width,
            "camera_height_px": ctx.height,
            "design_principle": (
                "camera -> physical occluder -> pedestrian"
            ),
        }

        print(
            (
                f"[SCENARIO] {self.name} | "
                f"occluder={occ.type_id} | "
                f"van_depth={actual_van_depth:.2f}m | "
                f"van_lat={actual_van_lat:.2f}m | "
                f"ped_depth={ped_depth:.2f}m | "
                f"ped_speed={self.ped_speed:.2f}m/s"
            ),
            flush=True,
        )

        return SceneActors(
            ped=ped,
            occluder=occ,
            tick=tick,
            hold=hold,
            footer=(
                "SCENARIO: RED — pedestrian crosses "
                "BEHIND PARKED VEHICLE | "
                "VISIBLE → OCCLUDED → EMERGES"
            ),
            meta=metadata,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, records, motion) -> dict:
        """
        Research-oriented quality gates.

        The scenario passes only if the recorded sequence demonstrates
        the intended temporal structure.

        We do NOT require every individual frame to satisfy a threshold.
        """

        if not records:
            return {
                "pass": False,
                "reason": "no_records",
            }

        # --------------------------------------------------------------
        # Basic measurements
        # --------------------------------------------------------------

        modal = [
            r["modal_area_px"]
            for r in records
        ]

        occ = [
            r["occlusion_ratio"]
            for r in records
        ]

        full_flags = [
            bool(r["fully_occluded"])
            for r in records
        ]

        cover_flags = [
            bool(r["amodal_covers_modal"])
            for r in records
        ]

        peak_occ = max(
            occ,
            default=0.0,
        )

        full_frames = sum(full_flags)

        cover_rate = (
            sum(cover_flags)
            / max(1, len(cover_flags))
        )

        # --------------------------------------------------------------
        # Establish early visibility reference
        # --------------------------------------------------------------

        quarter = max(
            1,
            len(records) // 4,
        )

        early = records[:quarter]
        late = records[-quarter:]

        early_visible = max(
            (
                r["modal_area_px"]
                for r in early
            ),
            default=0,
        )

        late_visible = max(
            (
                r["modal_area_px"]
                for r in late
            ),
            default=0,
        )

        # --------------------------------------------------------------
        # Detect meaningful occlusion interval
        # --------------------------------------------------------------

        strong_occ_indices = [
            i
            for i, r in enumerate(records)
            if r["occlusion_ratio"] >= 0.75
        ]

        severe_occ_indices = [
            i
            for i, r in enumerate(records)
            if r["occlusion_ratio"] >= 0.50
        ]

        has_strong_occlusion = (
            len(strong_occ_indices) >= 5
        )

        has_severe_occlusion = (
            len(severe_occ_indices) >= 8
        )

        # --------------------------------------------------------------
        # Temporal structure
        # --------------------------------------------------------------

        first_strong = (
            min(strong_occ_indices)
            if strong_occ_indices
            else None
        )

        last_strong = (
            max(strong_occ_indices)
            if strong_occ_indices
            else None
        )

        sustained_strong = (
            first_strong is not None
            and last_strong is not None
            and (last_strong - first_strong + 1) >= 5
        )

        # --------------------------------------------------------------
        # Motion quality
        # --------------------------------------------------------------

        pedestrian_moved = not motion[
            "ped_not_moving"
        ]

        pedestrian_no_teleport = not motion[
            "ped_teleport"
        ]

        occluder_stationary = (
            motion["occ_max_speed_mps"] < 0.20
        )

        # --------------------------------------------------------------
        # Spatial / visibility structure
        # --------------------------------------------------------------

        # We need meaningful evidence before and after the event.
        visible_before = early_visible >= 150
        visible_after = late_visible >= 120

        # Require a real loss of visibility, not just a tiny fluctuation.
        visibility_drop = (
            early_visible > 0
            and peak_occ >= 0.90
        )

        # --------------------------------------------------------------
        # Phase sanity
        # --------------------------------------------------------------

        phases = [
            str(r.get("phase", ""))
            for r in records
        ]

        phase_text = " ".join(
            p.upper()
            for p in phases
        )

        partial_seen = (
            "PARTIAL_OCCLUSION" in phase_text
        )

        severe_seen = (
            "SEVERE_OCCLUSION" in phase_text
        )

        # --------------------------------------------------------------
        # Gates
        # --------------------------------------------------------------

        gates = {
            "initial_visibility": visible_before,
            "final_visibility": visible_after,
            "meaningful_visibility_drop": visibility_drop,
            "severe_occlusion_seen": (
                has_severe_occlusion
                or severe_seen
            ),
            "strong_occlusion_seen": (
                has_strong_occlusion
            ),
            "sustained_occlusion": (
                sustained_strong
            ),
            "full_occlusion_frames": full_frames,
            "peak_occlusion_ratio": peak_occ,
            "amodal_cover_rate": cover_rate,
            "partial_phase_seen": partial_seen,
            "severe_phase_seen": severe_seen,
            "pedestrian_moved": pedestrian_moved,
            "pedestrian_no_teleport": pedestrian_no_teleport,
            "occluder_stationary": occluder_stationary,
            "pedestrian_mean_speed_mps": motion[
                "ped_mean_speed_mps"
            ],
        }

        gates["pass"] = all(
            [
                gates["initial_visibility"],
                gates["final_visibility"],
                gates["meaningful_visibility_drop"],
                gates["severe_occlusion_seen"],
                gates["strong_occlusion_seen"],
                gates["sustained_occlusion"],
                gates["pedestrian_moved"],
                gates["pedestrian_no_teleport"],
                gates["occluder_stationary"],
                gates["amodal_cover_rate"] >= 0.85,
            ]
        )

        return gates


# =============================================================================
# Family member: pedestrian emerges between two parked cars
# =============================================================================

class PedestrianEmergeBetweenParkedCars:
    """
    Research question
    -----------------
    How does pedestrian observability change as a VRU crosses BEHIND a
    pair of side-by-side parked vehicles (curb parking wall), moving from
    the open curb side into the open roadway?

    Occlusion mechanism
    -------------------
    Two parked cars share approximately the same depth and sit side-by-side
    in lateral. Together they form a wide silhouette. The pedestrian walks
    at greater depth (behind that silhouette), so line-of-sight is blocked
    while they traverse the cars, then clears on the roadway side.

      ego camera
          |
          v
      [CarL][CarR]      ← same depth, different lat (parked wall)
            \\
             ped path (greater depth)  curb(+lat) → roadway(-lat)

    Camera → parked-car wall → pedestrian.

    Visibility sequence
    -------------------
      SIDEWALK_VISIBLE → ENTERING_GAP / IN_GAP_OCCLUDED → ROAD_VISIBLE

    Why not a depth-wise bumper gap?
    --------------------------------
    A gap aligned with the camera optical axis is an OPEN corridor — the
    ped stays visible inside it. That fails the litmus test. Lateral
    traversal BEHIND a car wall creates real LOS occlusion.

    Controlled variables
    --------------------
      car_depth, car_lat (left car), pair_spacing, ped_speed, ped lats, seed
    """

    name = "pedestrian_emerge_between_parked_cars"

    narrative = (
        "Ego stopped at red. Two vehicles are parked side-by-side along "
        "the right curb, forming a wide silhouette. A pedestrian crosses "
        "behind them from curb to roadway — visible, then occluded by the "
        "parked wall, then reappearing in the road."
    )

    def __init__(
        self,
        ped_speed: float = 1.35,
        car_depth: float = 11.0,
        car_lat: float = -4.8,
        pair_spacing: float = 2.6,
        ped_depth_offset: float = 3.5,
        ped_start_lat: float = -7.5,
        ped_end_lat: float = 2.5,
    ):
        self.ped_speed = float(ped_speed)
        self.car_depth = float(car_depth)
        self.car_lat = float(car_lat)
        self.pair_spacing = float(pair_spacing)
        self.ped_depth_offset = float(ped_depth_offset)
        self.ped_start_lat = float(ped_start_lat)
        self.ped_end_lat = float(ped_end_lat)

    def setup(self, ctx) -> SceneActors:
        carla = ctx.carla
        random.seed(getattr(ctx, "seed", 11))

        ped_candidates = list(ctx.bp.filter("walker.pedestrian.*"))
        if not ped_candidates:
            raise RuntimeError("No pedestrian blueprints available")
        ped_bp = random.choice(ped_candidates)
        if ped_bp.has_attribute("is_invincible"):
            ped_bp.set_attribute("is_invincible", "true")

        preferred = (
            "vehicle.toyota.prius",
            "vehicle.audi.a2",
            "vehicle.nissan.patrol",
            "vehicle.mini.cooper_s",
            "vehicle.volkswagen.t2",
        )
        car_bp = _pick_vehicle_bp(ctx.bp, preferred)
        car_bp.set_attribute("role_name", "blindbench_parked_left")
        car_bp_r = _pick_vehicle_bp(ctx.bp, preferred[1:] + preferred[:1])
        try:
            car_bp_r.set_attribute("role_name", "blindbench_parked_right")
        except Exception:
            pass

        yaw = ctx.et.rotation.yaw
        car_left = None
        actual_depth = None
        left_lat = None
        for dep, lat in (
            (self.car_depth, self.car_lat),
            (self.car_depth, self.car_lat - 0.3),
            (self.car_depth + 0.4, self.car_lat),
            (self.car_depth - 0.4, self.car_lat),
        ):
            tf = carla.Transform(
                ctx.loc(dep, lat, ctx.road_z + 0.45),
                carla.Rotation(yaw=yaw),
            )
            car_left = ctx.world.try_spawn_actor(car_bp, tf)
            if car_left is not None:
                actual_depth, left_lat = dep, lat
                break
        if car_left is None:
            raise RuntimeError("Failed to spawn left parked car")
        park_left = _park_vehicle(carla, car_left)
        ctx.world.tick()

        right_lat = left_lat + self.pair_spacing
        car_right = None
        for dep, lat in (
            (actual_depth, right_lat),
            (actual_depth, right_lat + 0.3),
            (actual_depth + 0.3, right_lat),
            (actual_depth, right_lat - 0.2),
        ):
            tf = carla.Transform(
                ctx.loc(dep, lat, ctx.road_z + 0.45),
                carla.Rotation(yaw=yaw),
            )
            car_right = ctx.world.try_spawn_actor(car_bp_r, tf)
            if car_right is not None:
                right_lat = lat
                break
        if car_right is None:
            raise RuntimeError("Failed to spawn right parked car")
        park_right = _park_vehicle(carla, car_right)

        ped_depth = actual_depth + self.ped_depth_offset
        ped_start = ctx.loc(ped_depth, self.ped_start_lat, ctx.road_z + 1.05)
        ped_end = ctx.loc(ped_depth, self.ped_end_lat, ctx.road_z + 1.05)
        direction_x, direction_y = unit_dir(ped_start, ped_end)

        ped = None
        for lat0 in (
            self.ped_start_lat,
            self.ped_start_lat + 0.5,
            self.ped_start_lat - 0.5,
            self.ped_start_lat + 1.0,
        ):
            spawn = ctx.loc(ped_depth, lat0, ctx.road_z + 1.05)
            ped = ctx.world.try_spawn_actor(
                ped_bp,
                carla.Transform(spawn, carla.Rotation(yaw=yaw + 90.0)),
            )
            if ped is not None:
                ped_start = spawn
                break
        if ped is None:
            raise RuntimeError("Failed to spawn pedestrian")

        for _ in range(3):
            idle = carla.WalkerControl()
            idle.speed = 0.0
            idle.direction = carla.Vector3D(direction_x, direction_y, 0.0)
            ped.apply_control(idle)
            ctx.world.tick()

        state = {"entered_gap": False, "emerged": False, "last_phase": "SIDEWALK_APPROACH"}
        start_tf = carla.Transform(ped_start, carla.Rotation(yaw=yaw + 90.0))

        def hold():
            car_left.apply_control(park_left)
            car_right.apply_control(park_right)

        def on_record_start():
            # After Brain lock (which walks the ped for visibility), snap
            # back once to the designed start pose for the recorded trial.
            ped.set_transform(start_tf)
            state["entered_gap"] = False
            state["emerged"] = False

        def tick_motion():
            here = ped.get_location()
            remaining = dist3(here, ped_end)
            if remaining < 0.40:
                stop = carla.WalkerControl()
                stop.speed = 0.0
                stop.direction = carla.Vector3D(direction_x, direction_y, 0.0)
                ped.apply_control(stop)
                return
            control = carla.WalkerControl()
            control.direction = carla.Vector3D(direction_x, direction_y, 0.0)
            control.speed = self.ped_speed
            control.jump = False
            ped.apply_control(control)

        def estimate_phase():
            ped_box = _project_bbox(ctx, ped)
            if ped_box is None:
                return "SIDEWALK_APPROACH"
            overlap = 0.0
            ped_z = 0.5 * (ped_box[4] + ped_box[5])
            for actor in (car_left, car_right):
                box = _project_bbox(ctx, actor)
                if box is None:
                    continue
                if 0.5 * (box[4] + box[5]) < ped_z:
                    overlap = max(overlap, _bbox_overlap(ped_box, box))

            here = ped.get_location()
            rel = here - ctx.origin
            lateral = rel.x * ctx.right.x + rel.y * ctx.right.y

            if overlap >= 0.55:
                state["entered_gap"] = True
                return "IN_GAP_OCCLUDED"
            if overlap >= 0.18:
                state["entered_gap"] = True
                return "ENTERING_GAP"
            if state["entered_gap"] and lateral > left_lat + self.pair_spacing + 0.5:
                state["emerged"] = True
                return "ROAD_VISIBLE"
            if state["emerged"]:
                return "ROAD_VISIBLE"
            return "SIDEWALK_APPROACH"

        def tick(frame_i: int) -> str:
            tick_motion()
            phase = estimate_phase()
            state["last_phase"] = phase
            return phase

        metadata = {
            "experiment": "pedestrian_emerge_between_parked_cars",
            "occlusion_mechanism": "side_by_side_parked_wall",
            "family": "dynamic_vehicle_occlusion",
            "occluder_near_type": car_left.type_id,
            "occluder_far_type": car_right.type_id,
            "car_depth_m": actual_depth,
            "car_left_lateral_m": left_lat,
            "car_right_lateral_m": right_lat,
            "pair_spacing_m": right_lat - left_lat,
            "pedestrian_depth_m": ped_depth,
            "pedestrian_depth_offset_m": self.ped_depth_offset,
            "pedestrian_speed_mps": self.ped_speed,
            "pedestrian_start_lateral_m": self.ped_start_lat,
            "pedestrian_end_lateral_m": self.ped_end_lat,
            "design_principle": "camera -> parked-car wall -> pedestrian",
            "research_question": (
                "How does observability change as a VRU crosses behind "
                "side-by-side parked vehicles from curb into roadway?"
            ),
            "litmus": (
                "depth-wise bumper gap rejected — opens toward camera; "
                "side-by-side wall creates real LOS occlusion"
            ),
        }

        print(
            (
                f"[SCENARIO] {self.name} | L={car_left.type_id} R={car_right.type_id} "
                f"| depth={actual_depth:.2f}m lat=[{left_lat:.2f},{right_lat:.2f}] "
                f"ped_depth={ped_depth:.2f}m speed={self.ped_speed:.2f}"
            ),
            flush=True,
        )

        return SceneActors(
            ped=ped,
            occluder=car_left,
            tick=tick,
            hold=hold,
            on_record_start=on_record_start,
            footer=(
                "SCENARIO: EMERGE — ped crosses BEHIND parked-car WALL | "
                "CURB → OCCLUDED → ROAD"
            ),
            meta=metadata,
        )

    def evaluate(self, records, motion) -> dict:
        if not records:
            return {"pass": False, "reason": "no_records"}

        n = len(records)
        q = max(1, n // 4)
        early, mid, late = records[:q], records[q : 3 * q], records[-q:]

        early_vis = max((r["modal_area_px"] for r in early), default=0)
        late_vis = max((r["modal_area_px"] for r in late), default=0)
        mid_peak_occ = max((r["occlusion_ratio"] for r in mid), default=0.0)
        peak_occ = max((r["occlusion_ratio"] for r in records), default=0.0)

        strong_idx = [i for i, r in enumerate(records) if r["occlusion_ratio"] >= 0.70]
        full_frames = sum(1 for r in records if r["fully_occluded"])
        cover_rate = sum(1 for r in records if r["amodal_covers_modal"]) / max(1, n)

        phases = " ".join(str(r.get("phase", "")).upper() for r in records)
        gap_phase = any(p in phases for p in ("IN_GAP_OCCLUDED", "ENTERING_GAP"))
        road_phase = "ROAD_VISIBLE" in phases or "EMERGING" in phases

        sustained = (
            len(strong_idx) >= 5
            and (max(strong_idx) - min(strong_idx) + 1) >= 5
            if strong_idx
            else False
        )

        got_occlusion = peak_occ >= 0.85 or full_frames >= 5 or mid_peak_occ >= 0.75

        gates = {
            "early_sidewalk_visibility": early_vis >= 120,
            "late_road_visibility": late_vis >= 200,
            "meaningful_occlusion": got_occlusion,
            "sustained_occlusion": sustained or full_frames >= 5,
            "full_occlusion_frames": full_frames,
            "peak_occlusion_ratio": peak_occ,
            "gap_phase_seen": gap_phase,
            "road_phase_seen": road_phase,
            "early_modal_px": early_vis,
            "late_modal_px": late_vis,
            "amodal_cover_rate": cover_rate,
            "pedestrian_moved": not motion["ped_not_moving"],
            "pedestrian_no_teleport": not motion["ped_teleport"],
            "occluder_stationary": motion["occ_max_speed_mps"] < 0.20,
            "pedestrian_mean_speed_mps": motion["ped_mean_speed_mps"],
        }

        gates["pass"] = all(
            [
                gates["early_sidewalk_visibility"],
                gates["late_road_visibility"],
                gates["meaningful_occlusion"],
                gates["sustained_occlusion"],
                gates["road_phase_seen"],
                gates["pedestrian_moved"],
                gates["pedestrian_no_teleport"],
                gates["occluder_stationary"],
                gates["amodal_cover_rate"] >= 0.85,
            ]
        )
        return gates


# =============================================================================
# Family member: rainy crosswalk behind curbside delivery van
# =============================================================================

class DeliveryVanCrosswalkOcclusion:
    """
    Realistic BlindBench occlusion scenario (physics-gated).

    Story
    -----
    Ego is stopped at a red light. A large curbside delivery van sits
    ahead in the near curb lane. A pedestrian crosses the road *behind*
    that van (camera → van → pedestrian), disappears from modal view,
    then re-emerges into the open lane.

    Physics contract (from BlindBench motion audit + python_api.md)
    --------------------------------------------------------------
    - Vehicles: set_simulate_physics(True); hand_brake + brake hold.
    - Occluder must stay stationary: occ_max_speed_mps < 0.20.
    - Pedestrian walks via WalkerControl only (no set_transform mid-run).
    - No teleport: ped step/tick ≤ 1.2 m (Brain flags ped_teleport).
    - Human walking speed ≈ 1.2–1.4 m/s (default 1.30).
    - Depth order must be camera → occluder → pedestrian.
    - Visibility arc: VISIBLE → PARTIAL → SEVERE → FULL → EMERGE → VISIBLE.

    Measurement authority remains BlindBenchBrain (modal/amodal GT).
    """

    name = "delivery_van_crosswalk_occlusion"

    narrative = (
        "Ego at red. Curbside delivery van is a stationary physical "
        "occluder. Pedestrian crosses behind it at walking speed: "
        "visible → occluded → emerges. Rain-friendly geometry."
    )

    def __init__(
        self,
        ped_speed: float = 1.30,
        van_depth: float = 14.0,
        van_lat: float = 4.2,
        ped_depth_offset: float = 3.2,
        # Start in open lane (visible for Brain ped-lock), walk behind van to far curb.
        ped_start_lat: float = -8.0,
        ped_end_lat: float = 8.0,
    ):
        # Clamp to human walking band (physics-plausible VRU speed).
        self.ped_speed = self._clamp(float(ped_speed), 1.10, 1.55)
        self.van_depth = float(van_depth)
        self.van_lat = float(van_lat)
        self.ped_depth_offset = float(ped_depth_offset)
        self.ped_start_lat = float(ped_start_lat)
        self.ped_end_lat = float(ped_end_lat)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def setup(self, ctx) -> SceneActors:
        carla = ctx.carla
        bp = ctx.world.get_blueprint_library()

        # Prefer a long/tall van so occlusion is a physical blocker, not a thin pole.
        occ_bp = _pick_vehicle_bp(
            bp,
            (
                "vehicle.mercedes.sprinter",
                "vehicle.ford.ambulance",
                "vehicle.volkswagen.t2",
                "vehicle.carlamotors.carlacola",
                "vehicle.tesla.cybertruck",
                "vehicle.*van*",
                "vehicle.*",
            ),
        )
        ped_bp = bp.filter("walker.pedestrian.*")[0]
        if ped_bp.has_attribute("is_invincible"):
            ped_bp.set_attribute("is_invincible", "false")

        yaw = ctx.et.rotation.yaw
        occ = None
        actual_depth = self.van_depth
        actual_lat = self.van_lat
        # Dense depth/lat grid — BlindBench ego-at-red pose varies by town.
        candidates = []
        for dep in (
            self.van_depth,
            self.van_depth + 1.0,
            self.van_depth + 2.0,
            self.van_depth - 1.0,
            self.van_depth + 3.5,
            10.0,
            16.0,
            18.0,
        ):
            for lat in (
                self.van_lat,
                self.van_lat + 0.6,
                self.van_lat - 0.6,
                self.van_lat + 1.2,
                self.van_lat - 1.2,
                2.5,
                5.5,
                -4.0,
                4.0,
            ):
                candidates.append((dep, lat))
        for dep, lat in candidates:
            loc = ctx.loc(dep, lat, ctx.road_z + 0.5)
            # Snap Z to driving surface when possible.
            try:
                wp = ctx.world.get_map().get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Any
                )
                if wp is not None:
                    loc.z = wp.transform.location.z + 0.4
            except Exception:
                pass
            tf = carla.Transform(loc, carla.Rotation(yaw=yaw))
            candidate = ctx.world.try_spawn_actor(occ_bp, tf)
            if candidate is not None:
                occ = candidate
                actual_depth, actual_lat = dep, lat
                break
        if occ is None:
            # Last resort: any large-ish vehicle at a map spawn near ego.
            near = sorted(
                ctx.world.get_map().get_spawn_points(),
                key=lambda sp: sp.location.distance(ctx.origin),
            )
            for sp in near[2:40]:
                candidate = ctx.world.try_spawn_actor(occ_bp, sp)
                if candidate is not None:
                    occ = candidate
                    # Recover approximate depth/lat in ego frame for ped placement.
                    rel = sp.location - ctx.origin
                    actual_depth = rel.x * ctx.fwd.x + rel.y * ctx.fwd.y
                    actual_lat = rel.x * ctx.right.x + rel.y * ctx.right.y
                    break
        if occ is None:
            raise RuntimeError("Failed to spawn delivery-van occluder")

        parked = _park_vehicle(carla, occ)
        ctx.world.tick()

        # Pedestrian BEHIND the van (greater depth) — required for real occlusion.
        ped_depth = actual_depth + self.ped_depth_offset
        ped_start = ctx.loc(ped_depth, self.ped_start_lat, ctx.road_z + 1.05)
        ped_end = ctx.loc(ped_depth, self.ped_end_lat, ctx.road_z + 1.05)
        direction_x, direction_y = unit_dir(ped_start, ped_end)

        ped = None
        for lat0 in (
            self.ped_start_lat,
            self.ped_start_lat + 0.4,
            self.ped_start_lat - 0.4,
            self.ped_start_lat + 0.8,
        ):
            spawn = ctx.loc(ped_depth, lat0, ctx.road_z + 1.05)
            ped = ctx.world.try_spawn_actor(
                ped_bp,
                carla.Transform(
                    spawn,
                    carla.Rotation(yaw=yaw + 90.0),
                ),
            )
            if ped is not None:
                ped_start = spawn
                break
        if ped is None:
            raise RuntimeError("Failed to spawn crosswalk pedestrian")

        # Settle feet on ground under physics (no teleports after record starts).
        for _ in range(3):
            idle = carla.WalkerControl()
            idle.speed = 0.0
            idle.direction = carla.Vector3D(direction_x, direction_y, 0.0)
            ped.apply_control(idle)
            ctx.world.tick()

        state = {
            "last_phase": "CURB_VISIBLE",
            "entered_occlusion": False,
            "severe_occlusion_seen": False,
            "emerged": False,
        }
        start_tf = carla.Transform(
            ped_start,
            carla.Rotation(yaw=yaw + 90.0),
        )

        def hold():
            occ.apply_control(parked)

        def on_record_start():
            ped.set_transform(start_tf)
            state["entered_occlusion"] = False
            state["severe_occlusion_seen"] = False
            state["emerged"] = False
            state["last_phase"] = "CURB_VISIBLE"

        def tick_motion():
            here = ped.get_location()
            remaining = dist3(here, ped_end)
            if remaining < 0.40:
                stop = carla.WalkerControl()
                stop.speed = 0.0
                stop.direction = carla.Vector3D(direction_x, direction_y, 0.0)
                ped.apply_control(stop)
                return
            control = carla.WalkerControl()
            control.direction = carla.Vector3D(direction_x, direction_y, 0.0)
            control.speed = self.ped_speed
            control.jump = False
            ped.apply_control(control)

        def estimate_phase():
            ped_box = _project_bbox(ctx, ped)
            occ_box = _project_bbox(ctx, occ)
            if ped_box is None or occ_box is None:
                return "CURB_VISIBLE"

            overlap = _bbox_overlap(ped_box, occ_box)
            ped_z = 0.5 * (ped_box[4] + ped_box[5])
            occ_z = 0.5 * (occ_box[4] + occ_box[5])
            # Ped must be farther than occluder in camera depth.
            if ped_z <= occ_z:
                return "DEPTH_ORDER_INVALID"

            if overlap >= 0.75:
                state["entered_occlusion"] = True
                state["severe_occlusion_seen"] = True
                return "SEVERE_OCCLUSION"
            if overlap >= 0.35:
                state["entered_occlusion"] = True
                return "PARTIAL_OCCLUSION"
            if state["entered_occlusion"] and overlap < 0.15:
                state["emerged"] = True
                return "EMERGE_VISIBLE"
            if state["emerged"]:
                return "LANE_VISIBLE"
            return "CURB_VISIBLE"

        def tick(frame_i: int) -> str:
            tick_motion()
            phase = estimate_phase()
            state["last_phase"] = phase
            return phase

        meta = {
            "experiment": "delivery_van_crosswalk_occlusion",
            "occlusion_mechanism": "parked_delivery_van",
            "depth_order": "camera → van → pedestrian",
            "pedestrian_speed_mps": self.ped_speed,
            "van_depth_m": actual_depth,
            "van_lat_m": actual_lat,
            "ped_depth_m": ped_depth,
            "ped_start_lat_m": self.ped_start_lat,
            "ped_end_lat_m": self.ped_end_lat,
            "physics": {
                "occluder_simulate_physics": True,
                "occluder_hand_brake": True,
                "ped_motion": "WalkerControl continuous (no mid-run teleport)",
                "ped_speed_band_mps": [1.10, 1.55],
                "occ_stationary_gate_mps": 0.20,
                "ped_teleport_gate_m_per_tick": 1.2,
            },
        }

        footer = (
            f"delivery_van depth={actual_depth:.1f}m lat={actual_lat:.1f}m | "
            f"ped_behind=+{self.ped_depth_offset:.1f}m @ {self.ped_speed:.2f}m/s"
        )

        return SceneActors(
            ped=ped,
            occluder=occ,
            tick=tick,
            footer=footer,
            meta=meta,
            hold=hold,
            on_record_start=on_record_start,
        )

    def evaluate(self, records, motion):
        n = len(records)
        if n < 60:
            return {"pass": False, "reason": "too_few_frames"}

        early = records[: max(20, n // 6)]
        late = records[-max(20, n // 6) :]

        def _vis(rows):
            return max((int(r.get("modal_area_px") or 0) for r in rows), default=0)

        early_visible = _vis(early)
        late_visible = _vis(late)
        peak_occ = max((float(r.get("occlusion_ratio") or 0.0) for r in records), default=0.0)
        full_frames = sum(1 for r in records if bool(r.get("full_occlusion")))

        strong = [i for i, r in enumerate(records) if float(r.get("occlusion_ratio") or 0) >= 0.75]
        severe = [i for i, r in enumerate(records) if float(r.get("occlusion_ratio") or 0) >= 0.50]
        cover = [
            bool(r.get("amodal_covers_modal"))
            for r in records
            if r.get("amodal_covers_modal") is not None
        ]
        cover_rate = (sum(cover) / len(cover)) if cover else 0.0

        phases = " ".join(str(r.get("phase", "")).upper() for r in records)
        sustained = bool(strong) and (max(strong) - min(strong) + 1) >= 5

        gates = {
            "initial_visibility": early_visible >= 150,
            "final_visibility": late_visible >= 120,
            "meaningful_visibility_drop": early_visible > 0 and peak_occ >= 0.90,
            "severe_occlusion_seen": len(severe) >= 8 or "SEVERE_OCCLUSION" in phases,
            "strong_occlusion_seen": len(strong) >= 5,
            "sustained_occlusion": sustained,
            "full_occlusion_frames": full_frames,
            "peak_occlusion_ratio": peak_occ,
            "amodal_cover_rate": cover_rate,
            "partial_phase_seen": "PARTIAL_OCCLUSION" in phases,
            "severe_phase_seen": "SEVERE_OCCLUSION" in phases,
            "pedestrian_moved": not motion["ped_not_moving"],
            "pedestrian_no_teleport": not motion["ped_teleport"],
            "occluder_stationary": motion["occ_max_speed_mps"] < 0.20,
            "pedestrian_mean_speed_mps": motion["ped_mean_speed_mps"],
            "depth_order_valid": "DEPTH_ORDER_INVALID" not in phases,
        }
        gates["pass"] = all(
            [
                gates["initial_visibility"],
                gates["final_visibility"],
                gates["meaningful_visibility_drop"],
                gates["severe_occlusion_seen"],
                gates["strong_occlusion_seen"],
                gates["sustained_occlusion"],
                gates["pedestrian_moved"],
                gates["pedestrian_no_teleport"],
                gates["occluder_stationary"],
                gates["depth_order_valid"],
                gates["amodal_cover_rate"] >= 0.85,
            ]
        )
        return gates


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "BlindBench scenario family — occlusion experiments "
            "(scenario causes; Brain measures)."
        )
    )

    ap.add_argument(
        "--scenario",
        choices=("parked_van", "emerge_gap", "delivery_van"),
        default="emerge_gap",
        help="Scenario family member to run.",
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--host",
        default="127.0.0.1",
    )

    ap.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--frames",
        type=int,
        default=280,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    ap.add_argument(
        "--height",
        type=int,
        default=720,
    )

    ap.add_argument(
        "--fov",
        type=float,
        default=70.0,
    )

    ap.add_argument(
        "--dt",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--ped-speed",
        type=float,
        default=1.35,
    )

    ap.add_argument(
        "--van-depth",
        type=float,
        default=14.0,
        help="parked_van: occluder depth (m).",
    )

    ap.add_argument(
        "--van-lat",
        type=float,
        default=4.0,
        help="parked_van: occluder lateral (m).",
    )

    ap.add_argument(
        "--ped-depth-offset",
        type=float,
        default=3.0,
        help="parked_van: ped depth behind van (m).",
    )

    ap.add_argument(
        "--car-depth",
        type=float,
        default=11.0,
        help="emerge_gap: parked cars depth (m).",
    )

    ap.add_argument(
        "--car-lat",
        type=float,
        default=-4.8,
        help="emerge_gap: left parked car lateral (m).",
    )

    ap.add_argument(
        "--gap",
        type=float,
        default=2.6,
        help="emerge_gap: lateral spacing between parked cars (m).",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=11,
    )

    ap.add_argument(
        "--town",
        default="",
    )

    args = ap.parse_args()

    brain = BlindBenchBrain(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        fov=args.fov,
        dt=args.dt,
        frames=args.frames,
        town=args.town,
        seed=args.seed,
    )

    if args.scenario == "parked_van":
        scenario = ParkedVanPedestrianOcclusion(
            ped_speed=args.ped_speed,
            van_depth=args.van_depth,
            van_lat=args.van_lat,
            ped_depth_offset=args.ped_depth_offset,
        )
    elif args.scenario == "delivery_van":
        scenario = DeliveryVanCrosswalkOcclusion(
            ped_speed=args.ped_speed,
            van_depth=args.van_depth,
            van_lat=args.van_lat,
            ped_depth_offset=args.ped_depth_offset,
        )
    else:
        scenario = PedestrianEmergeBetweenParkedCars(
            ped_speed=args.ped_speed,
            car_depth=args.car_depth,
            car_lat=args.car_lat,
            pair_spacing=args.gap,
        )

    brain.run(
        scenario,
        Path(args.out),
    )


if __name__ == "__main__":
    main()

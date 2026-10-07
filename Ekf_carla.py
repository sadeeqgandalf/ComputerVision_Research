#!/usr/bin/env python3
"""EKF localization from the ego vehicle's point of view.

The filter never reads ground truth. Truth is used only to simulate the
sensors (wheel speed, IMU gyro, range-bearing to signs the car can see)
and to score the belief.

    python Ekf_carla.py --mock
    python Ekf_carla.py --steps 1200 --vehicles 15 --walkers 8 --record

Python 3.7 compatible (CARLA 0.9.14 venv).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

MAX_UPDATES = 12         # landmarks fused per tick (nearest first)
RAY_CAP = 16             # line-of-sight rays cast per tick
P_LO, P_HI = 0.025, 0.975
P_INSIDE_2SIGMA_2D = 1.0 - math.exp(-2.0)   # P(chi2_2 <= 4) = 0.8647


@dataclass
class Cfg:
    dt: float = 0.05
    steps: int = 600
    alphas: Tuple[float, float, float, float] = (0.05, 0.01, 0.02, 0.05)
    sigma_gyro: float = 0.02          # rad/s, written onto the CARLA IMU
    sigma_r: float = 0.4              # m
    sigma_phi: float = math.radians(2.0)
    sigma0_xy: float = 0.5            # m, initial belief
    sigma0_th: float = math.radians(5.0)
    model_std_long: float = 0.0       # m per step, along-track motion-model error (R_t)
    model_std_lat: float = 0.0        # m per step, cross-track (sideslip) motion-model error
    ref_offset: Optional[float] = None   # m from actor origin along heading; None = rear axle
    weather: str = "clear"
    town: str = ""                       # e.g. "Town05"; empty keeps the loaded map
    max_range: float = 45.0
    fov: float = math.radians(90.0)
    cam_w: int = 960
    cam_h: int = 540
    seed: int = 0
    vehicles: int = 15
    walkers: int = 8
    host: str = "127.0.0.1"
    port: int = 2000
    record: bool = False


# ---------------------------------------------------------------------------
# Angles and chi-square
# ---------------------------------------------------------------------------


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    log_pref = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_pref)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(log_pref) * h


def chi2_cdf(x: float, dof: int) -> float:
    return _gammp(0.5 * dof, 0.5 * x)


@lru_cache(maxsize=None)
def chi2_ppf(p: float, dof: int) -> float:
    """Chi-square quantile by bisection on the exact CDF."""
    lo, hi = 0.0, max(1.0, float(dof))
    while chi2_cdf(hi, dof) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def mean_nees_band(n: int, dim: int = 3) -> Tuple[float, float]:
    """95% band for the mean of n NEES values if they were independent: chi2(dim*n)/n."""
    dof = dim * n
    if dof <= 300:
        return chi2_ppf(P_LO, dof) / n, chi2_ppf(P_HI, dof) / n
    # Wilson-Hilferty; relative error below 1e-3 above 300 dof (checked in tests).
    k = float(dof)
    c = 2.0 / (9.0 * k)
    z = 1.959963984540054
    lo = k * (1.0 - c - z * math.sqrt(c)) ** 3
    hi = k * (1.0 - c + z * math.sqrt(c)) ** 3
    return lo / n, hi / n


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


def control_cov(v: float, w: float, cfg: Cfg) -> np.ndarray:
    """Slide-32 M_t, plus the IMU gyro variance on omega."""
    a1, a2, a3, a4 = cfg.alphas
    var_v = (a1 * abs(v) + a2 * abs(w)) ** 2
    var_w = (a3 * abs(v) + a4 * abs(w)) ** 2 + cfg.sigma_gyro ** 2
    return np.diag([max(var_v, 1e-12), max(var_w, 1e-12)])


def noisy_control(v_true: float, w_in: float, cfg: Cfg, rng: np.random.Generator,
                  gyro_already_noisy: bool) -> np.ndarray:
    """Draw u from the same model as M_t. The CARLA IMU already carries sigma_gyro."""
    a1, a2, a3, a4 = cfg.alphas
    std_v = a1 * abs(v_true) + a2 * abs(w_in)
    var_w = (a3 * abs(v_true) + a4 * abs(w_in)) ** 2
    if not gyro_already_noisy:
        var_w += cfg.sigma_gyro ** 2
    return np.array([
        v_true + rng.normal(0.0, std_v),
        w_in + rng.normal(0.0, math.sqrt(var_w)),
    ])


def model_cov(phi: float, cfg: Cfg) -> np.ndarray:
    """R_t: body-frame (along, across) model error rotated to the world frame at chord heading phi."""
    R = np.zeros((3, 3))
    if cfg.model_std_long == 0.0 and cfg.model_std_lat == 0.0:
        return R
    c, s = math.cos(phi), math.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    R[:2, :2] = rot @ np.diag([cfg.model_std_long ** 2, cfg.model_std_lat ** 2]) @ rot.T
    return R


def body_residual(gt_prev: np.ndarray, gt: np.ndarray, v: float, dt: float) -> Tuple[float, float, float]:
    """One-step error of g(u, x) on truth, with omega taken from the true yaw change.

    Returns (along, across, phi): the position residual gt - g expressed in the
    chord frame, and the chord heading used for that frame.
    """
    w = wrap(float(gt[2] - gt_prev[2])) / dt
    pred = motion(gt_prev, np.array([v, w]), dt)[0]
    phi = float(gt_prev[2]) + 0.5 * w * dt
    dx, dy = float(gt[0] - pred[0]), float(gt[1] - pred[1])
    c, s = math.cos(phi), math.sin(phi)
    return c * dx + s * dy, -s * dx + c * dy, phi


def fit_reference_offset(poses: np.ndarray, dt: float) -> Tuple[float, float]:
    """Least-squares point on the car's long axis with zero sideways motion.

    `poses` are (x, y, theta) of any body point on the axis, one per tick. Moving
    the reference d metres forward changes each step's cross-track displacement
    by 2 d sin(w dt / 2), so the residual is linear in d. Returns (d, rms residual
    at d) in metres per step.
    """
    a = []
    k = []
    for p0, p1 in zip(poses[:-1], poses[1:]):
        dth = wrap(float(p1[2] - p0[2]))
        phi = float(p0[2]) + 0.5 * dth
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        a.append(-math.sin(phi) * dx + math.cos(phi) * dy)
        k.append(2.0 * math.sin(0.5 * dth))
    a_arr = np.asarray(a)
    k_arr = np.asarray(k)
    denom = float(k_arr @ k_arr)
    if denom < 1e-10:
        raise ValueError("no rotation in the calibration data; the offset is unobservable")
    d = -float(a_arr @ k_arr) / denom
    return d, float(np.sqrt(np.mean((a_arr + d * k_arr) ** 2)))


# ---------------------------------------------------------------------------
# Frame. CARLA is left-handed (y right, yaw clockwise). The filter is the
# right-handed unicycle from the slides: (x, -y, -yaw), omega_f = -omega_c.
# ---------------------------------------------------------------------------


def to_filter(x: float, y: float, yaw: float) -> np.ndarray:
    return np.array([x, -y, wrap(-yaw)], dtype=float)


def to_carla_xy(xy: Sequence[float]) -> Tuple[float, float]:
    return float(xy[0]), float(-xy[1])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _sinc_and_derivative(h: float) -> Tuple[float, float]:
    """S(h) = sin(h)/h and S'(h), stable at h = 0."""
    if abs(h) < 1e-4:
        h2 = h * h
        return 1.0 - h2 / 6.0 + h2 * h2 / 120.0, -h / 3.0 + h * h2 / 30.0
    s = math.sin(h)
    return s / h, (h * math.cos(h) - s) / (h * h)


def motion(mu: np.ndarray, u: np.ndarray, dt: float,
           G: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None):
    """Velocity motion model g(u, x) with G = dg/dx and V = dg/du.

    Slide 32 writes x' = x + (v/w)(sin(th + w dt) - sin th). With
    sin(A + 2h) - sin A = 2 cos(A + h) sin h and h = w dt / 2 this is exactly
    x' = x + v dt S(h) cos(th + h), which has no division by w and reduces to
    the straight-line model at w = 0.
    """
    v = float(u[0])
    w = float(u[1])
    x, y, th = float(mu[0]), float(mu[1]), float(mu[2])
    h = 0.5 * w * dt
    S, dS = _sinc_and_derivative(h)
    phi = th + h
    c = math.cos(phi)
    s = math.sin(phi)
    step = v * dt * S

    if G is None:
        G = np.empty((3, 3))
    if V is None:
        V = np.empty((3, 2))
    G[0, 0], G[0, 1], G[0, 2] = 1.0, 0.0, -step * s
    G[1, 0], G[1, 1], G[1, 2] = 0.0, 1.0, step * c
    G[2, 0], G[2, 1], G[2, 2] = 0.0, 0.0, 1.0

    half_vdt2 = 0.5 * v * dt * dt
    V[0, 0], V[0, 1] = dt * S * c, half_vdt2 * (dS * c - S * s)
    V[1, 0], V[1, 1] = dt * S * s, half_vdt2 * (dS * s + S * c)
    V[2, 0], V[2, 1] = 0.0, dt

    mu_bar = np.array([x + step * c, y + step * s, wrap(th + w * dt)])
    return mu_bar, G, V


def meas_model(mu: np.ndarray, landmark: np.ndarray):
    """Range-bearing h(x, m) and H = dh/dx (2x3)."""
    dx = float(landmark[0] - mu[0])
    dy = float(landmark[1] - mu[1])
    q = dx * dx + dy * dy
    if q < 1e-9:
        raise ValueError("landmark coincides with the pose; bearing is undefined")
    sq = math.sqrt(q)
    zhat = np.array([sq, wrap(math.atan2(dy, dx) - float(mu[2]))])
    H = np.array([
        [-dx / sq, -dy / sq, 0.0],
        [dy / q, -dx / q, -1.0],
    ])
    return zhat, H


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------


@dataclass
class UpdateResult:
    nu: np.ndarray       # innovation (2k,), bearings wrapped
    nis: float           # nu^T S^-1 nu, ~ chi2(2k) when consistent
    dof: int


class EKF:
    """State (x, y, theta). Control (v, omega). Joint range-bearing update."""

    def __init__(self, mu0: np.ndarray, Sigma0: np.ndarray, cfg: Cfg):
        self.mu = np.asarray(mu0, dtype=float).copy()
        self.Sigma = np.asarray(Sigma0, dtype=float).copy()
        self.cfg = cfg
        self._G = np.empty((3, 3))
        self._V = np.empty((3, 2))
        self._I = np.eye(3)
        cap = 2 * MAX_UPDATES
        self._H = np.empty((cap, 3))
        self._zhat = np.empty(cap)
        self._qdiag = np.empty(cap)
        self._qdiag[0::2] = cfg.sigma_r ** 2
        self._qdiag[1::2] = cfg.sigma_phi ** 2

    def predict(self, u: np.ndarray, dt: float) -> None:
        phi = float(self.mu[2]) + 0.5 * float(u[1]) * dt
        mu_bar, G, V = motion(self.mu, u, dt, self._G, self._V)
        M = control_cov(float(u[0]), float(u[1]), self.cfg)
        self.mu = mu_bar
        self.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T + model_cov(phi, self.cfg)

    def update(self, z: np.ndarray, landmarks: np.ndarray) -> UpdateResult:
        """All visible landmarks in one correction, linearized at the same mu_bar.

        With block-diagonal Q this is the exact EKF update for the stacked
        measurement; sequential updates would relinearize between landmarks.
        """
        landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
        k = min(int(landmarks.shape[0]), MAX_UPDATES)
        if k == 0:
            return UpdateResult(np.zeros(0), 0.0, 0)
        m = 2 * k
        z = np.asarray(z, dtype=float).ravel()[:m]
        H = self._H[:m]
        zhat = self._zhat[:m]
        for i in range(k):
            zh, Hi = meas_model(self.mu, landmarks[i])
            zhat[2 * i: 2 * i + 2] = zh
            H[2 * i: 2 * i + 2] = Hi
        q = self._qdiag[:m]

        nu = z - zhat
        nu[1::2] = wrap_arr(nu[1::2])
        HS = H @ self.Sigma                       # (m, 3)
        S = HS @ H.T
        S[np.diag_indices(m)] += q
        X = np.linalg.solve(S, np.column_stack([HS, nu]))
        K = X[:, :3].T                            # Sigma H^T S^-1
        nis = float(nu @ X[:, 3])

        self.mu = self.mu + K @ nu
        self.mu[2] = wrap(float(self.mu[2]))
        IKH = self._I - K @ H
        self.Sigma = IKH @ self.Sigma @ IKH.T + (K * q) @ K.T
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        return UpdateResult(nu.copy(), nis, m)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def pose_error(mu: np.ndarray, gt: np.ndarray) -> Tuple[float, float, np.ndarray]:
    e = mu - gt
    e[2] = wrap(float(e[2]))
    return math.hypot(float(e[0]), float(e[1])), float(e[2]), e


def nees(e: np.ndarray, Sigma: np.ndarray) -> float:
    return float(e @ np.linalg.solve(Sigma, e))


def sigma_xy(Sigma: np.ndarray) -> float:
    """RMS radius of the xy belief, sqrt(tr Sigma_xy)."""
    return math.sqrt(max(float(Sigma[0, 0] + Sigma[1, 1]), 0.0))


def inside_2sigma(e_xy: np.ndarray, Sigma: np.ndarray) -> bool:
    """Truth inside the 2-sigma xy ellipse (Mahalanobis^2 <= 4)."""
    return float(e_xy @ np.linalg.solve(Sigma[:2, :2], e_xy)) <= 4.0


def ellipse_points(mu: np.ndarray, Sigma: np.ndarray, n: int = 32) -> np.ndarray:
    """2-sigma xy ellipse, (n, 2), in the filter frame."""
    vals, vecs = np.linalg.eigh(Sigma[:2, :2])
    axes = 2.0 * np.sqrt(np.maximum(vals, 0.0))
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=0)
    return ((vecs * axes) @ circle).T + mu[:2]


class Residuals:
    """One-step motion-model residuals on truth. Their std calibrates R_t."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(3)      # along, across, heading
        self.s2 = np.zeros(3)

    def add(self, along: float, across: float, heading: float) -> None:
        r = np.array([along, across, heading])
        self.n += 1
        self.s += r
        self.s2 += r * r

    @property
    def mean(self) -> np.ndarray:
        return self.s / max(self.n, 1)

    @property
    def std(self) -> np.ndarray:
        """Sample standard deviation (n - 1)."""
        if self.n < 2:
            return np.zeros(3)
        var = (self.s2 - self.s * self.s / self.n) / (self.n - 1)
        return np.sqrt(np.maximum(var, 0.0))


class Stats:
    """Running consistency metrics over the whole run."""

    def __init__(self):
        self.n = 0
        self.pos2 = 0.0
        self.yaw2 = 0.0
        self.nees_sum = 0.0
        self.nees_in = 0
        self.inside = 0
        self.nis_n = 0
        self.nis_norm_sum = 0.0
        self.nis_in = 0
        self.nees_lo = chi2_ppf(P_LO, 3)
        self.nees_hi = chi2_ppf(P_HI, 3)
        self.residuals = Residuals()
        self.fitted_ref_offset = float("nan")
        self.fitted_std_lat = float("nan")

    def add(self, pos_err: float, yaw_err: float, nees_val: float, inside: bool,
            upd: UpdateResult) -> Tuple[bool, float, float]:
        self.n += 1
        self.pos2 += pos_err * pos_err
        self.yaw2 += yaw_err * yaw_err
        self.nees_sum += nees_val
        self.nees_in += int(self.nees_lo <= nees_val <= self.nees_hi)
        self.inside += int(inside)
        nis_lo = nis_hi = float("nan")
        nis_ok = False
        if upd.dof:
            nis_lo = chi2_ppf(P_LO, upd.dof)
            nis_hi = chi2_ppf(P_HI, upd.dof)
            nis_ok = nis_lo <= upd.nis <= nis_hi
            self.nis_n += 1
            self.nis_norm_sum += upd.nis / upd.dof
            self.nis_in += int(nis_ok)
        return nis_ok, nis_lo, nis_hi

    @property
    def rmse_pos(self) -> float:
        return math.sqrt(self.pos2 / max(self.n, 1))

    @property
    def rmse_yaw(self) -> float:
        return math.sqrt(self.yaw2 / max(self.n, 1))

    @property
    def nees_mean(self) -> float:
        return self.nees_sum / max(self.n, 1)

    @property
    def frac_nees_in(self) -> float:
        return self.nees_in / max(self.n, 1)

    @property
    def frac_inside(self) -> float:
        return self.inside / max(self.n, 1)

    @property
    def nis_norm_mean(self) -> float:
        return self.nis_norm_sum / self.nis_n if self.nis_n else float("nan")

    @property
    def frac_nis_in(self) -> float:
        return self.nis_in / self.nis_n if self.nis_n else float("nan")


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    z: np.ndarray                  # (2k,) range, bearing
    xy: np.ndarray                 # measured landmarks, filter frame (k, 2)
    xyz_carla: np.ndarray          # measured landmarks, CARLA xyz (k, 3)
    occluded_xyz: np.ndarray       # in range and FOV but blocked, CARLA xyz
    n_cand: int                    # in range and FOV before line of sight
    ids: List[int] = field(default_factory=list)   # map index of each measured landmark


@dataclass
class Box3D:
    kind: str                      # "vehicle" | "walker"
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]   # image segments, near-clipped
    distance: float                # m, ego to actor
    vis_px: int                    # pixels of this actor in the instance mask
    box2d: Optional[Tuple[int, int, int, int]]   # tight x0, y0, x1, y1 from the mask

    @property
    def visible(self) -> bool:
        return self.box2d is not None


# CARLA BoundingBox.get_world_vertices order -> the 12 box edges.
BOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (0, 4), (4, 5),
             (5, 1), (5, 7), (7, 6), (6, 4), (6, 2), (7, 3))
NEAR_PLANE = 0.3                   # m, camera near plane for clipping
# CARLA 0.9.14 semantic tags (CityScapes numbering)
TAGS_WALKER = (12, 13)             # pedestrian, rider
TAGS_VEHICLE = (14, 15, 16, 17, 18, 19)   # car, truck, bus, train, motorcycle, bicycle
MIN_VIS_PX = 25


def project_edges(cam: np.ndarray, focal: float, w: int, h: int,
                  edges=BOX_EDGES, near: float = NEAR_PLANE):
    """Project box edges given corners in camera coords (depth, right, up).

    Edges crossing the near plane are cut at depth = near instead of dropped,
    so a box partly behind the camera still draws its visible part.
    """
    out = []
    for a, b in edges:
        pa, pb = cam[a], cam[b]
        da, db = pa[0], pb[0]
        if da < near and db < near:
            continue
        if da < near:
            pa = pa + (near - da) / (db - da) * (pb - pa)
        elif db < near:
            pb = pb + (near - db) / (da - db) * (pa - pb)
        out.append(((0.5 * w + focal * pa[1] / pa[0], 0.5 * h - focal * pa[2] / pa[0]),
                    (0.5 * w + focal * pb[1] / pb[0], 0.5 * h - focal * pb[2] / pb[0])))
    return out


def _bgra(image) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))


def depth_to_vis(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Planar depth (m) to an RGB image: near = bright, log scale over 1..100 m."""
    if depth is None:
        return None
    d = np.clip(depth[::2, ::2], 1.0, 100.0)
    g = (255.0 * (1.0 - np.log(d) / math.log(100.0))).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def _empty_obs(n_cand: int = 0) -> Observation:
    return Observation(np.zeros(0), np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3)), n_cand)


def _gate(lm_xy: np.ndarray, gt: np.ndarray, cfg: Cfg, min_range: float):
    dx = lm_xy[:, 0] - gt[0]
    dy = lm_xy[:, 1] - gt[1]
    r = np.hypot(dx, dy)
    b = wrap_arr(np.arctan2(dy, dx) - gt[2])
    mask = (r > min_range) & (r <= cfg.max_range) & (np.abs(b) <= 0.5 * cfg.fov)
    idx = np.flatnonzero(mask)
    return idx[np.argsort(r[idx])], r, b


def _measure(r: np.ndarray, b: np.ndarray, keep: Sequence[int], cfg: Cfg,
             rng: np.random.Generator) -> np.ndarray:
    k = len(keep)
    z = np.empty(2 * k)
    z[0::2] = r[keep] + rng.normal(0.0, cfg.sigma_r, k)
    z[1::2] = wrap_arr(b[keep] + rng.normal(0.0, cfg.sigma_phi, k))
    return z


class MockWorld:
    """Unicycle driven by the same g(u, x). Right-handed frame, no occluders, no cameras."""

    gyro_already_noisy = False
    ref_offset = 0.0
    footprint = (0.0, 2.35, 1.0)
    seg = None
    depth = None
    lidar_xy = np.zeros((0, 2))

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.k = 0
        rng = np.random.default_rng(cfg.seed + 1)
        ang = rng.uniform(0.0, 2.0 * np.pi, 30)
        rad = rng.uniform(30.0, 70.0, 30)
        self.lm_filter = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        self.pose = np.zeros(3)

    def reset(self) -> np.ndarray:
        self.pose = np.array([0.0, -50.0, 0.0])
        self.k = 0
        return self.pose.copy()

    def step(self):
        dt = self.cfg.dt
        v = 8.0
        w = 8.0 / 50.0 + 0.08 * math.sin(0.4 * self.k * dt)
        self.pose = motion(self.pose, np.array([v, w]), dt)[0]
        self.k += 1
        return self.pose.copy(), v, w

    def observe(self, gt: np.ndarray, rng: np.random.Generator) -> Observation:
        idx, r, b = _gate(self.lm_filter, gt, self.cfg, 1.0)
        if idx.size == 0:
            return _empty_obs()
        keep = idx[:MAX_UPDATES].tolist()
        z = _measure(r, b, keep, self.cfg, rng)
        return Observation(z, self.lm_filter[keep], np.zeros((len(keep), 3)),
                           np.zeros((0, 3)), int(idx.size), keep)

    def camera_image(self):
        return None

    def project(self, xy_filter, z: float = 0.0):
        return None

    def project_carla(self, x: float, y: float, z: float):
        return None

    def boxes(self) -> List[Box3D]:
        return []

    def close(self) -> None:
        pass


class CarlaWorld:
    """Hero on autopilot. Town signs and lights are the map. IMU + wheel speed are the control."""

    gyro_already_noisy = True

    def __init__(self, cfg: Cfg):
        import carla
        self.carla = carla
        self.cfg = cfg
        self.actors: List = []
        self.controllers: List = []
        self.sensors = {}
        self.queues = {}
        self.cam = None
        self.ego = None
        self.frame = -1
        self.image = None
        self.seg = None
        self.depth = None
        self.inst_id = None
        self.inst_tag = None
        self._ego_inst: set = set()
        self.lidar_xy = np.zeros((0, 2))
        self.gyro_z = 0.0
        self.lm_filter = np.zeros((0, 2))
        self.lm_carla = np.zeros((0, 3))
        self.ref_offset = 0.0
        self.footprint = (0.0, 2.35, 1.0)
        self._v_prev = 0.0
        self._cam_ready = False
        self._origin = np.zeros(3)
        self._basis = np.eye(3)       # rows: forward, right, up
        self._focal = cfg.cam_w / (2.0 * math.tan(0.5 * cfg.fov))
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        if cfg.town and not self.world.get_map().name.endswith(cfg.town):
            print("loading", cfg.town, "(was %s)" % self.world.get_map().name, flush=True)
            self.client.set_timeout(180.0)
            self.world = self.client.load_world(cfg.town)
            self.client.set_timeout(30.0)
        self.original = self.world.get_settings()
        self.tm = self.client.get_trafficmanager(8000)
        labels = carla.CityObjectLabel
        self._occluders = set()
        for name in ("Vehicles", "Pedestrians", "Buildings", "Vegetation",
                     "Walls", "Fences", "Bridge", "Static", "Dynamic"):
            if hasattr(labels, name):
                self._occluders.add(getattr(labels, name))

    # --- sensors ---------------------------------------------------------

    def _attach(self, name: str, bp, transform) -> None:
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        q: queue.Queue = queue.Queue()
        sensor.listen(q.put)
        self.sensors[name] = sensor
        self.queues[name] = q

    def _wait(self, name: str, frame: int):
        """Block until the sensor delivers this exact frame; older frames are dropped."""
        q = self.queues[name]
        while True:
            try:
                data = q.get(timeout=5.0)
            except queue.Empty:
                raise RuntimeError("sensor %s did not deliver frame %d" % (name, frame))
            if data.frame >= frame:
                return data

    def _tick(self) -> None:
        self.frame = self.world.tick()
        data = {name: self._wait(name, self.frame) for name in self.sensors}
        self.gyro_z = float(data["imu"].gyroscope.z)

        image = data["rgb"]
        self.image = _bgra(image)[:, :, 2::-1].copy()
        tf = image.transform
        f, r, u = tf.get_forward_vector(), tf.get_right_vector(), tf.get_up_vector()
        self._origin = np.array([tf.location.x, tf.location.y, tf.location.z])
        self._basis = np.array([[f.x, f.y, f.z], [r.x, r.y, r.z], [u.x, u.y, u.z]])
        self._cam_ready = True

        seg = data["seg"]
        seg.convert(self.carla.ColorConverter.CityScapesPalette)
        self.seg = _bgra(seg)[:, :, 2::-1].copy()

        inst = _bgra(data["inst"])
        self.inst_tag = inst[:, :, 2].copy()                             # semantic tag
        self.inst_id = inst[:, :, 1].astype(np.int32) + 256 * inst[:, :, 0].astype(np.int32)

        bgra = _bgra(data["depth"]).astype(np.float32)
        # CARLA depth encoding: (R + 256 G + 65536 B) / (2^24 - 1) * 1000 m, planar depth.
        self.depth = (bgra[:, :, 2] + 256.0 * bgra[:, :, 1] + 65536.0 * bgra[:, :, 0]) \
            * (1000.0 / 16777215.0)

        lidar = data["lidar"]
        pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
        ltf = lidar.transform
        yaw = math.radians(ltf.rotation.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        xw = ltf.location.x + pts[:, 0] * cy - pts[:, 1] * sy
        yw = ltf.location.y + pts[:, 0] * sy + pts[:, 1] * cy
        keep = pts[:, 2] > -1.6          # drop road returns (sensor ~1.9 m above ground)
        self.lidar_xy = np.stack([xw[keep], -yw[keep]], axis=1)   # filter frame

    def _pose_filter(self) -> np.ndarray:
        """Pose of the reference point: the point on the long axis with no sideways
        velocity, which is the point the velocity motion model describes."""
        tf = self.ego.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        x = tf.location.x + self.ref_offset * math.cos(yaw)
        y = tf.location.y + self.ref_offset * math.sin(yaw)
        return to_filter(x, y, yaw)

    def _rear_axle_offset(self) -> float:
        """Signed distance along the heading from the actor origin to the rear-axle centre."""
        wheels = self.ego.get_physics_control().wheels     # FL, FR, RL, RR; positions in cm
        rear = np.array([[w.position.x, w.position.y] for w in wheels[2:4]]).mean(axis=0) / 100.0
        tf = self.ego.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        return float((rear[0] - tf.location.x) * math.cos(yaw) + (rear[1] - tf.location.y) * math.sin(yaw))

    # --- setup -----------------------------------------------------------

    def _load_landmarks(self) -> None:
        carla = self.carla
        pts = []
        for obj in self.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns):
            c = obj.bounding_box.location
            pts.append((c.x, c.y, c.z))
        for light in self.world.get_actors().filter("traffic.traffic_light*"):
            boxes = light.get_light_boxes() if hasattr(light, "get_light_boxes") else []
            if boxes:
                c = np.mean([[bb.location.x, bb.location.y, bb.location.z] for bb in boxes], axis=0)
                pts.append((float(c[0]), float(c[1]), float(c[2])))
            else:
                loc = light.get_location()
                pts.append((loc.x, loc.y, loc.z + 3.0))
        if not pts:
            raise RuntimeError("no traffic signs or lights in this town; the map m is empty")
        xyz = np.asarray(pts, dtype=float)
        self.lm_carla = xyz
        self.lm_filter = np.stack([xyz[:, 0], -xyz[:, 1]], axis=1)
        print("landmarks", len(pts), flush=True)

    def _spawn_ego(self, spawn_points) -> None:
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        for sp in spawn_points:
            self.ego = self.world.try_spawn_actor(bp, sp)
            if self.ego is not None:
                self.actors.append(self.ego)
                return
        raise RuntimeError("could not spawn the hero at any spawn point")

    def _spawn_sensors(self) -> None:
        """RGB, semantic masks and depth share one camera pose, so every pixel lines up."""
        carla = self.carla
        lib = self.world.get_blueprint_library()
        cam_tf = carla.Transform(carla.Location(x=1.4, z=1.5), carla.Rotation(pitch=-8.0))
        for name, bp_id in (("rgb", "sensor.camera.rgb"),
                            ("seg", "sensor.camera.semantic_segmentation"),
                            ("inst", "sensor.camera.instance_segmentation"),
                            ("depth", "sensor.camera.depth")):
            bp = lib.find(bp_id)
            bp.set_attribute("image_size_x", str(self.cfg.cam_w))
            bp.set_attribute("image_size_y", str(self.cfg.cam_h))
            bp.set_attribute("fov", "%.4f" % math.degrees(self.cfg.fov))
            self._attach(name, bp, cam_tf)
        self.cam = self.sensors["rgb"]

        lidar_bp = lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("range", "70")
        lidar_bp.set_attribute("points_per_second", "320000")
        lidar_bp.set_attribute("rotation_frequency", "%.4f" % (1.0 / self.cfg.dt))
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-30")
        self._attach("lidar", lidar_bp, carla.Transform(carla.Location(z=1.9)))

        imu_bp = lib.find("sensor.other.imu")
        imu_bp.set_attribute("noise_gyro_stddev_z", "%.6f" % self.cfg.sigma_gyro)
        imu_bp.set_attribute("noise_gyro_bias_z", "0.0")
        self._attach("imu", imu_bp, carla.Transform())

    def _spawn_traffic(self, spawn_points) -> None:
        carla = self.carla
        lib = self.world.get_blueprint_library()
        rng = random.Random(self.cfg.seed)
        vehicles = [bp for bp in lib.filter("vehicle.*")
                    if int(bp.get_attribute("number_of_wheels")) == 4]
        n = 0
        for sp in spawn_points:
            if n >= self.cfg.vehicles:
                break
            bp = rng.choice(vehicles)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue
            actor.set_autopilot(True, self.tm.get_port())
            self.actors.append(actor)
            n += 1

        walker_bps = lib.filter("walker.pedestrian.*")
        ctrl_bp = lib.find("controller.ai.walker")
        for _ in range(3 * self.cfg.walkers):
            if len(self.controllers) >= self.cfg.walkers:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                break
            bp = rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue
            self.actors.append(walker)
            ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), walker)
            self.controllers.append(ctrl)
        self.world.tick()
        for ctrl in self.controllers:
            ctrl.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4)
        print("traffic vehicles", n, "walkers", len(self.controllers), flush=True)

    def _gyro_check(self) -> None:
        """Drive one curve; the IMU yaw rate must match the yaw change in sign and scale (rad/s)."""
        carla = self.carla
        yaw_prev = math.radians(self.ego.get_transform().rotation.yaw)
        dyaw = 0.0
        integ = 0.0
        for _ in range(300):
            self.ego.apply_control(carla.VehicleControl(throttle=0.45, steer=0.3))
            self._tick()
            yaw = math.radians(self.ego.get_transform().rotation.yaw)
            dyaw += wrap(yaw - yaw_prev)
            yaw_prev = yaw
            integ += self.gyro_z * self.cfg.dt
            if abs(dyaw) > 0.6:
                break
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        ratio = integ / dyaw if abs(dyaw) > 1e-9 else float("nan")
        print("gyro check  dyaw %.4f rad  integral(gyro_z dt) %.4f rad  ratio %.3f"
              % (dyaw, integ, ratio), flush=True)
        if abs(dyaw) < 0.2:
            raise RuntimeError("hero did not turn during the gyro check (dyaw=%.4f rad)" % dyaw)
        if not 0.8 <= ratio <= 1.25:
            raise RuntimeError(
                "IMU gyro z disagrees with CARLA yaw (ratio %.3f); expected +1 in rad/s" % ratio
            )

        rear = self._rear_axle_offset()
        if self.cfg.ref_offset is not None:
            self.ref_offset = float(self.cfg.ref_offset)
            source = "given"
        else:
            self.ref_offset = rear
            source = "geometric rear axle"
        print("reference point %.3f m from actor origin (%s); rear axle at %.3f m"
              % (self.ref_offset, source, rear), flush=True)
        bb = self.ego.bounding_box
        self.footprint = (float(bb.location.x) - self.ref_offset, float(bb.extent.x), float(bb.extent.y))

    def reset(self) -> np.ndarray:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.cfg.dt
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.cfg.seed)
        presets = {
            "clear": "ClearNoon", "cloudy": "CloudyNoon", "rain": "HardRainNoon",
            "sunset": "ClearSunset", "night": "ClearNight",
        }
        if self.cfg.weather in presets:
            self.world.set_weather(getattr(self.carla.WeatherParameters, presets[self.cfg.weather]))

        spawn_points = list(self.world.get_map().get_spawn_points())
        self._spawn_ego(spawn_points)
        self._spawn_sensors()
        self._load_landmarks()
        self._tick()
        # Only the hero exists yet: vehicle pixels at the bottom of the frame are its own hood.
        bottom = slice(int(0.8 * self.cfg.cam_h), None)
        hood = np.isin(self.inst_tag[bottom], TAGS_VEHICLE)
        self._ego_inst = set(np.unique(self.inst_id[bottom][hood]).tolist())
        self._gyro_check()
        self._spawn_traffic([sp for sp in spawn_points if sp.location.distance(self.ego.get_location()) > 10.0])
        self.ego.set_autopilot(True, self.tm.get_port())
        self._tick()
        self._v_prev = self._forward_speed()
        return self._pose_filter()

    # --- per tick --------------------------------------------------------

    def _forward_speed(self) -> float:
        """Planar speed along the heading. It is the same at every point on the car's
        long axis (rigid body), so the centre velocity gives the reference-point v."""
        vel = self.ego.get_velocity()
        fwd = self.ego.get_transform().get_forward_vector()
        norm = math.hypot(fwd.x, fwd.y)
        return (vel.x * fwd.x + vel.y * fwd.y) / norm if norm > 1e-9 else 0.0

    def step(self):
        """Truth pose and control over the last tick.

        Wheel odometry reports distance per interval, i.e. the mean speed over the
        tick; the trapezoid of the two end speeds is that mean to O(dt^2).
        """
        self._tick()
        gt = self._pose_filter()
        v_now = self._forward_speed()
        v = 0.5 * (self._v_prev + v_now)
        self._v_prev = v_now
        return gt, v, -self.gyro_z

    def observe(self, gt: np.ndarray, rng: np.random.Generator) -> Observation:
        idx, r, b = _gate(self.lm_filter, gt, self.cfg, 1.5)
        n_cand = int(idx.size)
        if n_cand == 0:
            return _empty_obs()
        carla = self.carla
        start = self.cam.get_transform().location
        keep: List[int] = []
        blocked: List[int] = []
        for i in idx[:RAY_CAP].tolist():
            end = carla.Location(x=float(self.lm_carla[i, 0]), y=float(self.lm_carla[i, 1]),
                                 z=float(self.lm_carla[i, 2]))
            if self._in_sight(start, end):
                keep.append(i)
                if len(keep) >= MAX_UPDATES:
                    break
            else:
                blocked.append(i)
        occluded = self.lm_carla[blocked] if blocked else np.zeros((0, 3))
        if not keep:
            obs = _empty_obs(n_cand)
            obs.occluded_xyz = occluded
            return obs
        z = _measure(r, b, keep, self.cfg, rng)
        return Observation(z, self.lm_filter[keep], self.lm_carla[keep], occluded, n_cand, keep)

    def _in_sight(self, start, end) -> bool:
        """Blocked if any occluder is hit before the landmark (0.8 m slack for the sign's own mesh)."""
        limit = start.distance(end) - 0.8
        for hit in self.world.cast_ray(start, end):
            if hit.label in self._occluders and hit.location.distance(start) < limit:
                return False
        return True

    # --- drawing helpers -------------------------------------------------

    def camera_image(self):
        return self.image

    def project(self, xy_filter, z: float = 0.0):
        x, y = to_carla_xy(xy_filter)
        return self.project_carla(x, y, z)

    def project_carla(self, x: float, y: float, z: float):
        """Pinhole projection. CARLA camera: x forward, y right, z up; image v points down."""
        if not self._cam_ready:
            return None
        d = np.array([x, y, z]) - self._origin
        depth, right, up = self._basis @ d
        if depth <= 0.5:
            return None
        return (0.5 * self.cfg.cam_w + self._focal * right / depth,
                0.5 * self.cfg.cam_h - self._focal * up / depth)

    def boxes(self, max_dist: float = 80.0) -> List[Box3D]:
        """Ground-truth 3D boxes of other actors, with visibility from the instance mask.

        CARLA 0.9.14 instance IDs are not actor IDs, so each actor is matched to the
        most frequent unclaimed instance of its class inside its projected box,
        nearest actor first. The 2D box is the tight extent of those mask pixels.
        """
        if not self._cam_ready or self.inst_id is None:
            return []
        W, H = self.cfg.cam_w, self.cfg.cam_h
        ego_loc = self.ego.get_location()
        cands = []
        for actor in self.actors:
            if actor.id == self.ego.id or not actor.is_alive:
                continue
            tf = actor.get_transform()
            dist = tf.location.distance(ego_loc)
            if dist <= max_dist:
                cands.append((dist, actor, tf))
        cands.sort(key=lambda c: c[0])

        used = set(self._ego_inst)
        out: List[Box3D] = []
        for dist, actor, tf in cands:
            P = np.array([[p.x, p.y, p.z] for p in actor.bounding_box.get_world_vertices(tf)])
            cam = (P - self._origin) @ self._basis.T          # depth, right, up
            edges = project_edges(cam, self._focal, W, H)
            if not edges:
                continue
            us = [p[0] for e in edges for p in e]
            vs = [p[1] for e in edges for p in e]
            x0, x1 = max(int(min(us)), 0), min(int(max(us)) + 1, W)
            y0, y1 = max(int(min(vs)), 0), min(int(max(vs)) + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue
            walker = actor.type_id.startswith("walker")
            tags = TAGS_WALKER if walker else TAGS_VEHICLE
            ids = self.inst_id[y0:y1, x0:x1]
            sel = np.isin(self.inst_tag[y0:y1, x0:x1], tags)
            box2d = None
            vis = 0
            if sel.any():
                vals, cnt = np.unique(ids[sel], return_counts=True)
                for j in np.argsort(-cnt):
                    if cnt[j] < MIN_VIS_PX:
                        break
                    iid = int(vals[j])
                    if iid in used:
                        continue
                    used.add(iid)
                    vis = int(cnt[j])
                    ys, xs = np.nonzero(sel & (ids == iid))
                    box2d = (x0 + int(xs.min()), y0 + int(ys.min()),
                             x0 + int(xs.max()), y0 + int(ys.max()))
                    break
            out.append(Box3D("walker" if walker else "vehicle", edges, dist, vis, box2d))
        return out

    def close(self) -> None:
        """Stop callbacks and autopilot first, destroy in one batch, then drop every handle."""
        carla = self.carla
        for sensor in self.sensors.values():
            if sensor.is_alive:
                sensor.stop()
        for ctrl in self.controllers:
            if ctrl.is_alive:
                ctrl.stop()
        port = self.tm.get_port()
        for actor in self.actors:
            if actor.is_alive and actor.type_id.startswith("vehicle."):
                actor.set_autopilot(False, port)
        ids = [a.id for a in list(self.sensors.values()) + self.controllers + self.actors if a.is_alive]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self.sensors.clear()
        self.queues.clear()
        self.controllers = []
        self.actors = []
        self.cam = None
        self.ego = None
        try:
            self.tm.set_synchronous_mode(False)
        finally:
            self.world.apply_settings(self.original)


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "tick", "t", "gt_x", "gt_y", "gt_th", "mu_x", "mu_y", "mu_th",
    "sxx", "syy", "stt", "sxy", "sxt", "syt",
    "pos_err", "yaw_err", "sigma_xy", "nees", "nees_in_band", "inside_2s",
    "nis", "nis_dof", "nis_in_band",
    "v", "omega", "n_visible", "n_occluded", "n_cand",
    "ms_predict", "ms_update", "ms_visibility",
    "rmse_pos", "rmse_yaw", "nees_mean", "frac_nees_in", "frac_inside", "nis_norm_mean",
    "n_actors_fov", "n_actors_visible", "n_lidar",
    "res_along", "res_across", "res_heading",
]


class CsvLog:
    def __init__(self, root: Path):
        self.dir = root / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ticks.csv"
        self._fh = self.path.open("w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(CSV_COLUMNS)
        print("log", self.path, flush=True)

    def write(self, row: Sequence[float]) -> None:
        self._w.writerow(row)

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def _fmt_len(m: float) -> str:
    if not math.isfinite(m):
        return "  n/a"
    return "%5.1f cm" % (100.0 * m) if abs(m) < 1.0 else "%5.2f m" % m


def _footprint(pose: np.ndarray, fp: Tuple[float, float, float]) -> List[Tuple[float, float]]:
    """Car outline corners in the filter frame. fp = (centre ahead of reference, half length, half width)."""
    off, hl, hw = fp
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    cx, cy = float(pose[0]) + off * c, float(pose[1]) + off * s
    return [(cx + a * c - b * s, cy + a * s + b * c) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _nice_step(target: float) -> float:
    for s in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        if s >= target:
            return s
    return 20.0


def _honesty_verdict(st: "Stats") -> Tuple[str, Tuple[int, int, int]]:
    """Plain reading of NEES: share of ticks inside the chi-square(3) 95% range."""
    if st.n < 40:
        return "warming up", (150, 150, 160)
    if st.frac_nees_in >= 0.90:
        return "honest", (40, 220, 90)
    if st.nees_mean > 3.0:
        return "overconfident", (240, 60, 60)
    return "too cautious", (240, 200, 60)


class Hud:
    """Pygame window. Drawing only."""

    W, H = 1920, 1080
    PANEL = 390
    GREEN = (40, 220, 90)
    RED = (240, 60, 60)
    GOLD = (240, 200, 60)
    BLUE = (60, 150, 255)
    GREY = (140, 140, 150)
    ORANGE = (255, 150, 40)
    MAGENTA = (230, 80, 230)
    LIDAR = (120, 170, 210)

    def __init__(self, cfg: Cfg):
        if not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("EKF ego view")
        mono = "dejavusansmono,menlo,consolas,monospace"
        self.font = pygame.font.SysFont(mono, 15)
        self.small = pygame.font.SysFont(mono, 13)
        self.cfg = cfg
        self.hist_n = int(20.0 / cfg.dt)
        self.err_hist: List[float] = []
        self.sig_hist: List[float] = []
        self.nees_hist: List[float] = []
        self.trail_gt: List[Tuple[float, float]] = []
        self.trail_mu: List[Tuple[float, float]] = []
        self.recorder = None

    def attach_log_dir(self, directory: Path) -> None:
        if not self.cfg.record:
            return
        import cv2
        path = str(directory / "hud.mp4")
        self.recorder = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                        1.0 / self.cfg.dt, (self.W, self.H))
        print("video", path, flush=True)

    def pump(self) -> bool:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return False
        return True

    def draw(self, view: dict) -> None:
        self.screen.fill((12, 14, 18))
        self._push_hist(view)
        self._panel(view)
        self._main(view)
        self.pg.display.flip()
        if self.recorder is not None:
            frame = self.pg.surfarray.array3d(self.screen).transpose(1, 0, 2)
            self.recorder.write(np.ascontiguousarray(frame[:, :, ::-1]))

    def _push_hist(self, view: dict) -> None:
        for hist, key in ((self.err_hist, "pos_err"), (self.sig_hist, "sigma_xy"),
                          (self.nees_hist, "nees")):
            hist.append(view[key])
            if len(hist) > self.hist_n:
                del hist[0]
        self.trail_gt.append((float(view["gt"][0]), float(view["gt"][1])))
        self.trail_mu.append((float(view["mu"][0]), float(view["mu"][1])))
        if len(self.trail_gt) > 400:
            del self.trail_gt[0]
            del self.trail_mu[0]

    def _text(self, text: str, x: int, y: int, color=(230, 230, 230), small: bool = False) -> int:
        surf = (self.small if small else self.font).render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 3

    HEAD = (120, 200, 255)
    DIM = (150, 150, 160)

    def _panel(self, v: dict) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (20, 22, 28), (0, 0, self.PANEL, self.H))
        clip = self.screen.get_clip()
        self.screen.set_clip(pg.Rect(0, 0, self.PANEL, self.H))
        st: Stats = v["stats"]
        x = 14
        y = 10
        y = self._text("EKF localization of the hero car", x, y, self.HEAD)
        y = self._text("t %6.1f s   speed %4.1f m/s   turn %+.2f rad/s"
                       % (v["t"], v["v"], v["omega"]), x, y, self.DIM, small=True)

        y = self._section("1. How far off is the estimate?", x, y + 6)
        y = self._text("position error      %s" % _fmt_len(v["pos_err"]), x, y)
        y = self._text("heading error       %+.2f deg" % math.degrees(v["yaw_err"]), x, y)
        y = self._text("claimed error (RMS) %s" % _fmt_len(v["sigma_xy"]), x, y, (80, 220, 220))
        y = self._text("RMS so far          %s   %.2f deg"
                       % (_fmt_len(st.rmse_pos), math.degrees(st.rmse_yaw)), x, y, self.DIM, small=True)

        y = self._section("2. Is it honest about its uncertainty?", x, y + 6)
        verdict, vcolor = _honesty_verdict(st)
        y = self._text("verdict             %s" % verdict, x, y, vcolor)
        ok = st.nees_lo <= v["nees"] <= st.nees_hi
        y = self._text("NEES now            %5.2f   (honest ~ 3)" % v["nees"], x, y,
                       self.GREEN if ok else self.RED)
        y = self._text("NEES average        %5.2f" % st.nees_mean, x, y)
        y = self._text("NEES in 95%% range   %5.1f %%  (want 95)" % (100.0 * st.frac_nees_in), x, y)
        y = self._text("truth inside 2-sig  %5.1f %%  (want %.1f)"
                       % (100.0 * st.frac_inside, 100.0 * P_INSIDE_2SIGMA_2D), x, y)
        y = self._text("NEES squares error/claimed error", x, y, self.DIM, small=True)
        y = self._text("and sums x, y, and heading.", x, y, self.DIM, small=True)

        y = self._section("3. What did it measure this tick?", x, y + 6)
        y = self._text("signs used %2d   blocked %2d   in view %2d"
                       % (v["n_vis"], v["n_occ"], v["n_cand"]), x, y)
        if v["nis_dof"]:
            y = self._text("surprise NIS/dof     %4.2f   (want 1.00)" % (v["nis"] / v["nis_dof"]), x, y,
                           self.GREEN if v["nis_ok"] else self.RED)
        else:
            y = self._text("no sign visible: predicting from wheels + gyro", x, y, self.GOLD, small=True)
        y = self._text("  sign   range   range miss  bearing miss", x, y, self.DIM, small=True)
        for lid, rng_m, (rr, bb) in v["meas_rows"][:4]:
            y = self._text("  #%-4d %5.1f m   %+5.2f m    %+5.2f deg"
                           % (lid, rng_m, rr, math.degrees(bb)), x, y, small=True)

        y = self._section("4. What the car sees", x, y + 6)
        boxes: List[Box3D] = v["boxes"]
        veh = [b for b in boxes if b.kind == "vehicle"]
        ped = [b for b in boxes if b.kind == "walker"]
        y = self._text("vehicles   %2d visible  %2d hidden" % (sum(b.visible for b in veh),
                       sum(not b.visible for b in veh)), x, y)
        y = self._text("people     %2d visible  %2d hidden" % (sum(b.visible for b in ped),
                       sum(not b.visible for b in ped)), x, y)
        near = min((b.distance for b in boxes if b.visible), default=float("nan"))
        y = self._text("nearest visible %5.1f m   lidar %5d pts" % (near, v["n_lidar"]), x, y, small=True)

        y = self._section("5. Motion model vs truth, per 50 ms", x, y + 6)
        s = st.residuals.std
        y = self._text("along %s   sideways %s" % (_fmt_len(s[0]), _fmt_len(s[1])), x, y, small=True)
        y = self._text("compute  predict %.2f ms  update %.2f ms" % (v["ms_p"], v["ms_u"]),
                       x, y, self.DIM, small=True)
        self.screen.set_clip(clip)

        self._error_inset(v, pg.Rect(12, self.H - 318, self.PANEL - 24, 306))

    def _section(self, title: str, x: int, y: int) -> int:
        self.pg.draw.line(self.screen, (50, 54, 64), (x, y), (self.PANEL - 14, y), 1)
        return self._text(title, x, y + 4, self.HEAD, small=True)

    def _error_inset(self, v: dict, rect) -> None:
        """Truth at the centre, EKF estimate and its 2-sigma ellipse, drawn to scale."""
        pg = self.pg
        pg.draw.rect(self.screen, (26, 28, 34), rect)
        pg.draw.rect(self.screen, (70, 74, 84), rect, 1)
        self._text("zoom on the car (to scale)", rect.x + 8, rect.y + 4, self.HEAD, small=True)
        plot = pg.Rect(rect.x + 6, rect.y + 24, rect.w - 12, rect.h - 50)
        gt, mu, S = v["gt"], v["mu"], v["Sigma"]
        e = mu[:2] - gt[:2]
        r2 = 2.0 * math.sqrt(max(float(np.linalg.eigvalsh(S[:2, :2]).max()), 0.0))
        half = max(1.25 * r2, 1.25 * float(np.hypot(*e)), 0.02)
        scale = min(plot.w, plot.h) / (2.0 * half)
        cx, cy = plot.centerx, plot.centery
        step = _nice_step(half / 2.0)
        n = int(half / step) + 2
        clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        for i in range(-n, n + 1):
            off = i * step * scale
            col = (60, 64, 74) if i else (80, 84, 96)
            pg.draw.line(self.screen, col, (cx + off, plot.top), (cx + off, plot.bottom), 1)
            pg.draw.line(self.screen, col, (plot.left, cy + off), (plot.right, cy + off), 1)
        ring = [(cx + (p[0] - gt[0]) * scale, cy - (p[1] - gt[1]) * scale) for p in ellipse_points(mu, S)]
        pg.draw.lines(self.screen, self.GOLD, True, ring, 2)
        m = (cx + e[0] * scale, cy - e[1] * scale)
        pg.draw.line(self.screen, (230, 230, 230), (cx, cy), m, 1)
        for (px, py), th, color in (((cx, cy), gt[2], self.GREEN), (m, mu[2], self.RED)):
            tip = (px + 34 * math.cos(th), py - 34 * math.sin(th))
            pg.draw.line(self.screen, color, (px, py), tip, 2)
            pg.draw.circle(self.screen, color, (int(px), int(py)), 5)
        self.screen.set_clip(clip)
        self._text("grid %s   green truth  red EKF  gold 2-sigma" % _fmt_len(step),
                   rect.x + 8, rect.bottom - 22, self.DIM, small=True)

    def _chart(self, rect, series, title, refs=()) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (28, 30, 36), rect, border_radius=4)
        self._text(title, rect.x + 6, rect.y + 2, (170, 170, 180), small=True)
        vals = [val for s, _ in series for val in s] + list(refs)
        if not vals or len(series[0][0]) < 2:
            return
        peak = max(max(vals), 1e-3) * 1.1
        plot = pg.Rect(rect.x + 6, rect.y + 20, rect.w - 12, rect.h - 26)
        for ref in refs:
            yy = plot.bottom - plot.h * ref / peak
            pg.draw.line(self.screen, (80, 80, 92), (plot.x, yy), (plot.right, yy), 1)
        for s, color in series:
            n = len(s)
            pts = [(plot.x + plot.w * i / (n - 1), plot.bottom - plot.h * val / peak)
                   for i, val in enumerate(s)]
            pg.draw.lines(self.screen, color, False, pts, 1)

    def _main(self, v: dict) -> None:
        pg = self.pg
        cw, ch = self.cfg.cam_w, self.cfg.cam_h
        origin = (self.PANEL + 16, 16)
        image = v["image"]
        if image is not None:
            self.screen.blit(pg.surfarray.make_surface(image.transpose(1, 0, 2)), origin)
            self._overlay(v, origin)
        else:
            rect = pg.Rect(origin[0], origin[1], cw, ch)
            pg.draw.rect(self.screen, (24, 28, 34), rect)
            self._draw_map(v, rect, scale=6.0)
        self._text("hero camera: 3D boxes from the instance mask, signs the EKF used (blue) or could not see (grey)",
                   origin[0] + 8, origin[1] + 6, (235, 235, 235), small=True)

        col_x = origin[0] + cw + 16
        col_w = self.W - col_x - 16
        thumb_h = int(round(col_w * ch / float(cw)))
        self._thumb(pg.Rect(col_x, 16, col_w, thumb_h), v["seg"], "semantic masks (CityScapes)")
        self._thumb(pg.Rect(col_x, 24 + thumb_h, col_w, thumb_h), v["depth_vis"],
                    "depth, log scale 1..100 m")

        self._legend(origin[0], origin[1] + ch + 8)
        top = origin[1] + ch + 52
        width = (cw - 12) // 2
        height = self.H - top - 16
        self._chart(pg.Rect(origin[0], top, width, height),
                    [(self.err_hist, (240, 240, 240)), (self.sig_hist, (80, 220, 220))],
                    "last 20 s: actual error (white) vs claimed error (cyan), m")
        self._chart(pg.Rect(origin[0] + width + 12, top, width, height),
                    [(self.nees_hist, self.GOLD)],
                    "last 20 s: NEES, honest ~ 3 (lower line); above upper line = overconfident",
                    refs=(3.0, v["stats"].nees_hi))
        map_top = 32 + 2 * thumb_h
        self._minimap(v, pg.Rect(col_x, map_top, col_w, self.H - map_top - 16))

    def _thumb(self, rect, image: Optional[np.ndarray], title: str) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (24, 28, 34), rect)
        if image is not None:
            surf = pg.surfarray.make_surface(image.transpose(1, 0, 2))
            self.screen.blit(pg.transform.scale(surf, (rect.w, rect.h)), rect.topleft)
        else:
            self._text("no camera in mock mode", rect.x + 12, rect.centery - 8, self.GREY, small=True)
        pg.draw.rect(self.screen, (70, 74, 84), rect, 1)
        self._text(title, rect.x + 8, rect.y + 6, (240, 240, 240), small=True)

    def _legend(self, x: int, y: int) -> None:
        items = (("true car", self.GREEN), ("EKF estimate", self.RED), ("2-sigma", self.GOLD),
                 ("sign used", self.BLUE), ("sign blocked", self.GREY),
                 ("vehicle", self.ORANGE), ("pedestrian", self.MAGENTA),
                 ("hidden actor", (110, 110, 120)), ("lidar", self.LIDAR))
        x0, right = x, x + self.cfg.cam_w
        for label, color in items:
            if x + 16 + self.small.size(label)[0] > right:
                x, y = x0, y + 18
            self.pg.draw.circle(self.screen, color, (x + 6, y + 8), 5)
            x = self._text_inline(label, x + 16, y) + 18

    def _text_inline(self, text: str, x: int, y: int) -> int:
        surf = self.small.render(text, True, (200, 200, 210))
        self.screen.blit(surf, (x, y))
        return x + surf.get_width()

    def _to_screen(self, uv, origin):
        return (int(origin[0] + uv[0]), int(origin[1] + uv[1]))

    def _overlay(self, v: dict, origin) -> None:
        pg = self.pg
        ox, oy = origin
        clip = self.screen.get_clip()
        self.screen.set_clip(pg.Rect(ox, oy, self.cfg.cam_w, self.cfg.cam_h))
        for box in sorted(v["boxes"], key=lambda b: -b.distance):
            if not box.visible:
                for (u0, v0), (u1, v1) in box.edges:
                    pg.draw.line(self.screen, (110, 110, 120), (ox + u0, oy + v0), (ox + u1, oy + v1), 1)
                continue
            color = self.MAGENTA if box.kind == "walker" else self.ORANGE
            for (u0, v0), (u1, v1) in box.edges:
                pg.draw.line(self.screen, color, (ox + u0, oy + v0), (ox + u1, oy + v1), 2)
            x0, y0, x1, y1 = box.box2d
            pg.draw.rect(self.screen, (255, 255, 255), (ox + x0, oy + y0, x1 - x0 + 1, y1 - y0 + 1), 1)
            label = "%s %.0f m" % ("person" if box.kind == "walker" else "car", box.distance)
            self._tag(label, ox + x0, oy + y0 - 17, color)
        for xyz in v["occluded_xyz"]:
            uv = v["project_carla"](*xyz)
            if uv is not None:
                c = self._to_screen(uv, origin)
                pg.draw.circle(self.screen, self.GREY, c, 8, 2)
                self._tag("blocked", c[0] + 10, c[1] - 8, self.GREY)
        for xyz, lid, rng_m in zip(v["meas_xyz"], v["meas_ids"], v["meas_range"]):
            uv = v["project_carla"](*xyz)
            if uv is None:
                continue
            c = self._to_screen(uv, origin)
            pg.draw.circle(self.screen, self.BLUE, c, 8, 2)
            pg.draw.circle(self.screen, self.BLUE, c, 2)
            self._tag("#%d %.0f m" % (lid, rng_m), c[0] + 10, c[1] - 8, self.BLUE)
        self.screen.set_clip(clip)

    def _tag(self, text: str, x: int, y: int, color) -> None:
        surf = self.small.render(text, True, (255, 255, 255))
        bg = self.pg.Rect(x - 2, y - 1, surf.get_width() + 4, surf.get_height() + 2)
        self.pg.draw.rect(self.screen, (0, 0, 0), bg)
        self.pg.draw.rect(self.screen, color, bg, 1)
        self.screen.blit(surf, (x, y))

    def _minimap(self, v: dict, rect) -> None:
        scale = 4.0
        self.pg.draw.rect(self.screen, (18, 20, 26), rect)
        self._draw_map(v, rect, scale=scale, lidar=True)
        self.pg.draw.rect(self.screen, (70, 74, 84), rect, 1)
        self._text("bird's-eye: lidar, sign map, true car vs EKF car", rect.x + 8, rect.y + 6,
                   (220, 220, 230), small=True)
        x0, yb = rect.x + 12, rect.bottom - 14
        self.pg.draw.line(self.screen, (230, 230, 230), (x0, yb), (x0 + 10 * scale, yb), 2)
        self._text_inline("10 m", int(x0 + 10 * scale + 6), yb - 8)

    def _draw_map(self, v: dict, rect, scale: float, lidar: bool = False) -> None:
        pg = self.pg
        gx, gy = float(v["gt"][0]), float(v["gt"][1])
        cx, cy = rect.centerx, rect.centery

        def pix(x, y):
            return (cx + (x - gx) * scale, cy - (y - gy) * scale)

        clip = self.screen.get_clip()
        self.screen.set_clip(rect)
        pts = v["lidar_xy"]
        if lidar and pts.shape[0]:
            px = (cx + (pts[:, 0] - gx) * scale).astype(int)
            py = (cy - (pts[:, 1] - gy) * scale).astype(int)
            ok = (px >= rect.left) & (px < rect.right) & (py >= rect.top) & (py < rect.bottom)
            px, py = px[ok], py[ok]
            if px.size:
                arr = pg.surfarray.pixels3d(self.screen)
                arr[px, py] = self.LIDAR
                del arr
        if len(self.trail_gt) >= 2:
            pg.draw.lines(self.screen, (40, 160, 70), False, [pix(*p) for p in self.trail_gt], 1)
            pg.draw.lines(self.screen, (180, 50, 50), False, [pix(*p) for p in self.trail_mu], 1)
        for xy in v["nearby"]:
            p = pix(xy[0], xy[1])
            pg.draw.circle(self.screen, (110, 120, 140), (int(p[0]), int(p[1])), 3, 1)
        m = pix(float(v["mu"][0]), float(v["mu"][1]))
        for xy, lid in zip(v["meas_xy"], v["meas_ids"]):
            p = pix(xy[0], xy[1])
            pg.draw.line(self.screen, self.BLUE, m, p, 1)
            pg.draw.circle(self.screen, self.BLUE, (int(p[0]), int(p[1])), 4)
            self._text_inline("#%d" % lid, int(p[0]) + 5, int(p[1]) - 7)
        fp = v["footprint"]
        pg.draw.polygon(self.screen, self.GREEN, [pix(*c) for c in _footprint(v["gt"], fp)], 0)
        pg.draw.polygon(self.screen, self.RED, [pix(*c) for c in _footprint(v["mu"], fp)], 2)
        self.screen.set_clip(clip)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.release()
        self.pg.quit()


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def run(world, cfg: Cfg, hud: Optional[Hud] = None, log_root: Optional[Path] = None) -> Stats:
    rng = np.random.default_rng(cfg.seed)
    try:
        gt = world.reset()
    except BaseException:
        if hud is not None:
            hud.close()
        world.close()
        raise
    Sigma0 = np.diag([cfg.sigma0_xy ** 2, cfg.sigma0_xy ** 2, cfg.sigma0_th ** 2])
    mu0 = gt + np.sqrt(np.diag(Sigma0)) * rng.standard_normal(3)
    mu0[2] = wrap(float(mu0[2]))
    ekf = EKF(mu0, Sigma0, cfg)
    stats = Stats()
    log = CsvLog(log_root or (Path.home() / "carla-runtime" / "ekf"))
    if hud is not None:
        hud.attach_log_dir(log.dir)
    gt_prev = gt.copy()
    truth: List[np.ndarray] = [gt.copy()]
    try:
        for k in range(cfg.steps):
            if hud is not None and not hud.pump():
                break
            gt, v_true, w_true = world.step()
            along, across, _ = body_residual(gt_prev, gt, v_true, cfg.dt)
            res_heading = wrap(float(gt[2] - gt_prev[2]) - w_true * cfg.dt)
            stats.residuals.add(along, across, res_heading)
            gt_prev = gt.copy()
            truth.append(gt_prev)
            t0 = time.perf_counter()
            obs = world.observe(gt, rng)
            t1 = time.perf_counter()
            u = noisy_control(v_true, w_true, cfg, rng, world.gyro_already_noisy)
            ekf.predict(u, cfg.dt)
            t2 = time.perf_counter()
            upd = ekf.update(obs.z, obs.xy)
            t3 = time.perf_counter()

            pos_err, yaw_err, e = pose_error(ekf.mu, gt)
            nees_val = nees(e, ekf.Sigma)
            inside = inside_2sigma(e[:2], ekf.Sigma)
            nis_ok, nis_lo, nis_hi = stats.add(pos_err, yaw_err, nees_val, inside, upd)
            band_lo, band_hi = mean_nees_band(stats.n)
            sig = sigma_xy(ekf.Sigma)
            S = ekf.Sigma
            boxes = world.boxes()
            n_box_vis = sum(1 for b in boxes if b.visible)
            log.write([
                k, (k + 1) * cfg.dt, gt[0], gt[1], gt[2], ekf.mu[0], ekf.mu[1], ekf.mu[2],
                S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[0, 2], S[1, 2],
                pos_err, yaw_err, sig, nees_val,
                int(stats.nees_lo <= nees_val <= stats.nees_hi), int(inside),
                upd.nis, upd.dof, int(nis_ok),
                u[0], u[1], obs.xy.shape[0], obs.occluded_xyz.shape[0], obs.n_cand,
                (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t1 - t0) * 1e3,
                stats.rmse_pos, stats.rmse_yaw, stats.nees_mean,
                stats.frac_nees_in, stats.frac_inside, stats.nis_norm_mean,
                len(boxes), n_box_vis, world.lidar_xy.shape[0],
                along, across, res_heading,
            ])
            if hud is not None:
                d = world.lm_filter - gt[:2]
                hud.draw({
                    "tick": k, "t": (k + 1) * cfg.dt, "stats": stats,
                    "gt": gt, "mu": ekf.mu, "Sigma": ekf.Sigma,
                    "pos_err": pos_err, "yaw_err": yaw_err, "sigma_xy": sig,
                    "inside": inside, "nees": nees_val,
                    "band_lo": band_lo, "band_hi": band_hi,
                    "nis": upd.nis, "nis_dof": upd.dof, "nis_ok": nis_ok,
                    "nis_lo": nis_lo, "nis_hi": nis_hi,
                    "v": float(u[0]), "omega": float(u[1]),
                    "n_vis": obs.xy.shape[0], "n_occ": obs.occluded_xyz.shape[0],
                    "n_cand": obs.n_cand,
                    "ms_p": (t2 - t1) * 1e3, "ms_u": (t3 - t2) * 1e3, "ms_v": (t1 - t0) * 1e3,
                    "residuals": list(zip(upd.nu[0::2].tolist(), upd.nu[1::2].tolist())),
                    "image": world.camera_image(),
                    "project": world.project, "project_carla": world.project_carla,
                    "meas_xy": obs.xy, "meas_xyz": obs.xyz_carla, "meas_ids": obs.ids,
                    "meas_range": obs.z[0::2].tolist(),
                    "meas_rows": list(zip(obs.ids, obs.z[0::2].tolist(),
                                          zip(upd.nu[0::2].tolist(), upd.nu[1::2].tolist()))),
                    "footprint": world.footprint,
                    "occluded_xyz": obs.occluded_xyz,
                    "nearby": world.lm_filter[np.hypot(d[:, 0], d[:, 1]) < 90.0],
                    "boxes": boxes, "seg": world.seg,
                    "depth_vis": depth_to_vis(world.depth),
                    "lidar_xy": world.lidar_xy, "n_lidar": world.lidar_xy.shape[0],
                })
            if k % 50 == 0:
                print("tick %5d  pos %6.3f m  yaw %6.2f deg  NEES %6.2f  mean %5.2f  NIS/dof %5.2f"
                      "  seen %2d  occ %2d"
                      % (k, pos_err, math.degrees(yaw_err), nees_val, stats.nees_mean,
                         stats.nis_norm_mean, obs.xy.shape[0], obs.occluded_xyz.shape[0]),
                      flush=True)
    finally:
        log.close()
        if hud is not None:
            hud.close()
        world.close()

    lo, hi = mean_nees_band(max(stats.n, 1))
    print("\nticks %d | pos RMSE %.3f m | yaw RMSE %.2f deg" % (
        stats.n, stats.rmse_pos, math.degrees(stats.rmse_yaw)))
    print("mean NEES %.2f (independent-tick band %.2f..%.2f; ticks are correlated, so this band is narrow)"
          % (stats.nees_mean, lo, hi))
    print("NEES in chi2(3) 95%% band %.1f %% | inside 2-sigma %.1f %% (expect %.1f) | NIS/dof %.2f | NIS in band %.1f %%"
          % (100 * stats.frac_nees_in, 100 * stats.frac_inside, 100 * P_INSIDE_2SIGMA_2D,
             stats.nis_norm_mean, 100 * stats.frac_nis_in))
    res = stats.residuals
    m, s = res.mean, res.std
    print("motion-model residual per step (n=%d, ref %.3f m): along mean %+.4f std %.4f m | across mean %+.4f"
          " std %.4f m | heading vs gyro std %.5f rad (gyro noise alone: %.5f)"
          % (res.n, world.ref_offset, m[0], s[0], m[1], s[1], s[2], cfg.sigma_gyro * cfg.dt))
    try:
        delta, rms_lat = fit_reference_offset(np.asarray(truth), cfg.dt)
    except ValueError:
        delta, rms_lat = 0.0, math.sqrt(m[1] ** 2 + s[1] ** 2)
    offset = world.ref_offset + delta
    print("zero-sideslip point on this drive: %.3f m from actor origin (used %.3f), cross-track rms there %.5f m/step"
          % (offset, world.ref_offset, rms_lat))
    print("calibration flags for the next run: --ref-offset %.3f --model-std-long %.4f --model-std-lat %.4f"
          % (offset, math.sqrt(m[0] ** 2 + s[0] ** 2), rms_lat), flush=True)
    stats.fitted_ref_offset = offset
    stats.fitted_std_lat = rms_lat
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="EKF localization from the ego vehicle")
    p.add_argument("--mock", action="store_true", help="numpy unicycle, no CARLA")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--vehicles", type=int, default=15)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--record", action="store_true", help="write hud.mp4 next to ticks.csv")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weather", default="clear",
                   choices=("clear", "cloudy", "rain", "sunset", "night", "keep"))
    p.add_argument("--model-std-long", type=float, default=0.0,
                   help="R_t along-track std per step (m); use the value printed by a calibration run")
    p.add_argument("--model-std-lat", type=float, default=0.0,
                   help="R_t cross-track std per step (m)")
    p.add_argument("--ref-offset", type=float, default=None,
                   help="state reference point, m from actor origin along heading (default: rear axle)")
    p.add_argument("--town", default="", help="load this map first, e.g. Town05")
    a = p.parse_args()
    cfg = Cfg(steps=a.steps, vehicles=a.vehicles, walkers=a.walkers,
              record=a.record, host=a.host, port=a.port, seed=a.seed, weather=a.weather,
              model_std_long=a.model_std_long, model_std_lat=a.model_std_lat,
              ref_offset=a.ref_offset, town=a.town)
    world = MockWorld(cfg) if a.mock else CarlaWorld(cfg)
    hud = None if a.no_hud else Hud(cfg)
    run(world, cfg, hud)


if __name__ == "__main__":
    main()

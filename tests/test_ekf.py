"""Checks every model in Ekf_carla.py against an independent reference.

    python -m unittest tests/test_ekf.py -v
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Ekf_carla as E  # noqa: E402

DT = 0.05


def slide_arc(mu, u, dt):
    """Slide 32, verbatim. Only valid for omega != 0."""
    x, y, th = mu
    v, w = u
    r = v / w
    return np.array([
        x + r * (math.sin(th + w * dt) - math.sin(th)),
        y + r * (math.cos(th) - math.cos(th + w * dt)),
        E.wrap(th + w * dt),
    ])


def slide_arc_jacobians(mu, u, dt):
    """Slide 32 G_t and V_t, verbatim."""
    _, _, th = mu
    v, w = u
    s0, c0 = math.sin(th), math.cos(th)
    s1, c1 = math.sin(th + w * dt), math.cos(th + w * dt)
    G = np.array([[1, 0, (v / w) * (c1 - c0)],
                  [0, 1, (v / w) * (s1 - s0)],
                  [0, 0, 1]])
    V = np.array([[(s1 - s0) / w, v * (s0 - s1) / w ** 2 + v * dt * c1 / w],
                  [(c0 - c1) / w, -v * (c0 - c1) / w ** 2 + v * dt * s1 / w],
                  [0, dt]])
    return G, V


def numeric_jacobian(f, x, eps=1e-6, angle_rows=()):
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        d = np.zeros_like(x)
        d[j] = eps
        diff = f(x + d) - f(x - d)
        for r in angle_rows:
            diff[r] = E.wrap(diff[r])
        J[:, j] = diff / (2 * eps)
    return J


class TestMotionModel(unittest.TestCase):
    CASES = [
        (np.array([3.0, -2.0, 0.4]), np.array([8.0, 0.3])),
        (np.array([0.0, 0.0, -2.9]), np.array([12.0, -0.9])),
        (np.array([10.0, 5.0, 3.1]), np.array([5.0, 0.02])),
        (np.array([-4.0, 1.0, 1.2]), np.array([-2.0, 0.5])),   # reversing
    ]

    def test_equals_slide_arc_formula(self):
        for mu, u in self.CASES:
            got = E.motion(mu, u, DT)[0]
            ref = slide_arc(mu, u, DT)
            np.testing.assert_allclose(got[:2], ref[:2], atol=1e-12)
            self.assertAlmostEqual(E.wrap(got[2] - ref[2]), 0.0, places=12)

    def test_jacobians_equal_slide_jacobians(self):
        for mu, u in self.CASES:
            _, G, V = E.motion(mu, u, DT)
            Gs, Vs = slide_arc_jacobians(mu, u, DT)
            np.testing.assert_allclose(G, Gs, atol=1e-11)
            np.testing.assert_allclose(V, Vs, atol=1e-9)

    def test_straight_line_limit(self):
        mu = np.array([1.0, 2.0, 0.7])
        v = 9.0
        got, G, V = E.motion(mu, np.array([v, 0.0]), DT)
        np.testing.assert_allclose(
            got, [1.0 + v * DT * math.cos(0.7), 2.0 + v * DT * math.sin(0.7), 0.7], atol=1e-14)
        s, c = math.sin(0.7), math.cos(0.7)
        np.testing.assert_allclose(G, [[1, 0, -v * DT * s], [0, 1, v * DT * c], [0, 0, 1]], atol=1e-14)
        np.testing.assert_allclose(
            V, [[DT * c, -0.5 * v * DT ** 2 * s], [DT * s, 0.5 * v * DT ** 2 * c], [0, DT]], atol=1e-14)

    def test_continuous_across_omega_zero(self):
        mu = np.array([0.0, 0.0, 0.3])
        ref = E.motion(mu, np.array([10.0, 0.0]), DT)
        for w in (1e-12, -1e-9, 1e-6, -1e-5, 3e-4):
            got = E.motion(mu, np.array([10.0, w]), DT)
            for a, b in zip(got, ref):
                np.testing.assert_allclose(a, b, atol=5e-4 * abs(w) / 1e-4 + 1e-12)

    def test_numeric_jacobians_all_regimes(self):
        for w in (0.0, 1e-7, 1e-3, 0.4, -1.3):
            mu = np.array([2.0, -1.0, 0.9])
            u = np.array([7.0, w])
            _, G, V = E.motion(mu, u, DT)
            Gn = numeric_jacobian(lambda x: E.motion(x, u, DT)[0], mu, angle_rows=(2,))
            Vn = numeric_jacobian(lambda uu: E.motion(mu, uu, DT)[0], u, angle_rows=(2,))
            np.testing.assert_allclose(G, Gn, atol=1e-7)
            np.testing.assert_allclose(V, Vn, atol=1e-7)


class TestMeasurementModel(unittest.TestCase):
    def test_range_bearing(self):
        mu = np.array([1.0, 1.0, math.pi / 2])
        zhat, _ = E.meas_model(mu, np.array([1.0, 11.0]))
        np.testing.assert_allclose(zhat, [10.0, 0.0], atol=1e-12)
        zhat, _ = E.meas_model(mu, np.array([-9.0, 1.0]))
        np.testing.assert_allclose(zhat, [10.0, math.pi / 2], atol=1e-12)

    def test_numeric_H(self):
        lm = np.array([14.0, -3.0])
        for mu in (np.array([2.0, 5.0, 0.3]), np.array([-1.0, -8.0, -3.0]), np.array([20.0, 0.0, 2.5])):
            _, H = E.meas_model(mu, lm)
            Hn = numeric_jacobian(lambda x: E.meas_model(x, lm)[0], mu, angle_rows=(1,))
            np.testing.assert_allclose(H, Hn, atol=1e-7)


class TestUpdate(unittest.TestCase):
    def test_joint_update_equals_information_form(self):
        cfg = E.Cfg()
        mu = np.array([4.0, -2.0, 0.6])
        Sigma = np.array([[0.30, 0.05, 0.01], [0.05, 0.20, -0.02], [0.01, -0.02, 0.01]])
        lms = np.array([[20.0, 3.0], [10.0, -15.0], [30.0, 12.0]])
        rng = np.random.default_rng(3)
        z = []
        for lm in lms:
            zh, _ = E.meas_model(mu, lm)
            z += [zh[0] + rng.normal(0, 0.3), E.wrap(zh[1] + rng.normal(0, 0.03))]
        z = np.array(z)

        ekf = E.EKF(mu, Sigma, cfg)
        res = ekf.update(z, lms)

        H = np.vstack([E.meas_model(mu, lm)[1] for lm in lms])
        zhat = np.concatenate([E.meas_model(mu, lm)[0] for lm in lms])
        nu = z - zhat
        nu[1::2] = E.wrap_arr(nu[1::2])
        Q = np.diag([cfg.sigma_r ** 2, cfg.sigma_phi ** 2] * len(lms))
        P = np.linalg.inv(np.linalg.inv(Sigma) + H.T @ np.linalg.inv(Q) @ H)
        mu_ref = mu + P @ H.T @ np.linalg.inv(Q) @ nu
        S = H @ Sigma @ H.T + Q

        np.testing.assert_allclose(ekf.Sigma, P, atol=1e-12)
        np.testing.assert_allclose(ekf.mu[:2], mu_ref[:2], atol=1e-12)
        self.assertAlmostEqual(E.wrap(ekf.mu[2] - mu_ref[2]), 0.0, places=12)
        self.assertAlmostEqual(res.nis, float(nu @ np.linalg.solve(S, nu)), places=10)
        self.assertEqual(res.dof, 6)

    def test_bearing_residual_wraps(self):
        cfg = E.Cfg()
        mu = np.array([0.0, 0.0, 0.0])
        ekf = E.EKF(mu, np.diag([0.1, 0.1, 0.01]), cfg)
        lm = np.array([-10.0, 1e-3])                         # bearing just under +pi
        zhat, _ = E.meas_model(mu, lm)
        z = np.array([zhat[0], E.wrap(zhat[1] + 0.02)])      # crosses to -pi side
        res = ekf.update(z, lm[None, :])
        self.assertLess(abs(res.nu[1]), 0.03)

    def test_predict_only_grows_covariance(self):
        cfg = E.Cfg()
        ekf = E.EKF(np.zeros(3), np.diag([0.1, 0.1, 0.01]), cfg)
        before = np.linalg.eigvalsh(ekf.Sigma)
        ekf.predict(np.array([10.0, 0.2]), DT)
        after = np.linalg.eigvalsh(ekf.Sigma)
        self.assertGreater(np.linalg.det(ekf.Sigma), np.prod(before))
        self.assertTrue(np.all(after > 0))


class TestModelError(unittest.TestCase):
    def test_model_cov_rotates_body_axes(self):
        cfg = E.Cfg(model_std_long=0.3, model_std_lat=0.1)
        for phi in (0.0, 0.7, -2.2):
            R = E.model_cov(phi, cfg)
            along = np.array([math.cos(phi), math.sin(phi)])
            across = np.array([-math.sin(phi), math.cos(phi)])
            self.assertAlmostEqual(float(along @ R[:2, :2] @ along), 0.09, places=12)
            self.assertAlmostEqual(float(across @ R[:2, :2] @ across), 0.01, places=12)
            self.assertAlmostEqual(float(along @ R[:2, :2] @ across), 0.0, places=12)
            np.testing.assert_allclose(R[2], 0.0)

    def test_predict_adds_exactly_R(self):
        mu, S0, u = np.array([1.0, 2.0, 0.4]), np.diag([0.2, 0.3, 0.01]), np.array([9.0, 0.3])
        a = E.EKF(mu, S0, E.Cfg())
        b = E.EKF(mu, S0, E.Cfg(model_std_long=0.2, model_std_lat=0.05))
        a.predict(u, DT)
        b.predict(u, DT)
        phi = 0.4 + 0.5 * 0.3 * DT
        np.testing.assert_allclose(b.Sigma - a.Sigma, E.model_cov(phi, b.cfg), atol=1e-14)

    def test_residual_zero_on_exact_unicycle(self):
        x = np.array([3.0, -1.0, 0.8])
        for w in (0.0, 0.25, -0.6):
            x1 = E.motion(x, np.array([11.0, w]), DT)[0]
            along, across, _ = E.body_residual(x, x1, 11.0, DT)
            self.assertAlmostEqual(along, 0.0, places=12)
            self.assertAlmostEqual(across, 0.0, places=12)

    def test_residual_measures_sideslip(self):
        x = np.array([0.0, 0.0, 0.5])
        x1 = E.motion(x, np.array([10.0, 0.0]), DT)[0]
        x1[:2] += 0.02 * np.array([-math.sin(0.5), math.cos(0.5)])   # 2 cm to the left
        along, across, _ = E.body_residual(x, x1, 10.0, DT)
        self.assertAlmostEqual(along, 0.0, places=12)
        self.assertAlmostEqual(across, 0.02, places=12)


class TestReferencePoint(unittest.TestCase):
    def test_recovers_zero_sideslip_point(self):
        # The no-slip point follows the unicycle exactly; the logged body point sits
        # d_true metres behind it on the long axis, so it slides sideways in turns.
        d_true = -0.75
        ref = np.array([0.0, 0.0, 0.2])
        poses = []
        for k in range(120):
            w = 0.35 + 0.2 * math.sin(0.1 * k)
            ref = E.motion(ref, np.array([6.0, w]), DT)[0]
            poses.append([ref[0] - d_true * math.cos(ref[2]),
                          ref[1] - d_true * math.sin(ref[2]), ref[2]])
        d, rms = E.fit_reference_offset(np.array(poses), DT)
        self.assertAlmostEqual(d, d_true, places=9)
        self.assertLess(rms, 1e-10)

    def test_straight_driving_is_unobservable(self):
        poses = np.array([[0.1 * k, 0.0, 0.0] for k in range(20)])
        with self.assertRaises(ValueError):
            E.fit_reference_offset(poses, DT)


class TestBoxProjection(unittest.TestCase):
    F, W, H = 480.0, 960, 540

    def test_box_in_front_projects_all_edges(self):
        # Unit cube 10 m ahead, centred: all 12 edges, symmetric about the image centre.
        c = np.array([[10 + dx, dy, dz] for dx in (-.5, .5) for dy in (-.5, .5) for dz in (-.5, .5)])
        edges = E.project_edges(c, self.F, self.W, self.H, edges=_cube_edges())
        self.assertEqual(len(edges), 12)
        us = [p[0] for e in edges for p in e]
        self.assertAlmostEqual((min(us) + max(us)) / 2, self.W / 2, places=9)

    def test_edge_crossing_camera_is_clipped_not_dropped(self):
        cam = np.array([[-2.0, 1.0, -1.0], [8.0, 1.0, -1.0]])
        (p0, p1), = E.project_edges(cam, self.F, self.W, self.H, edges=((0, 1),))
        # The clipped end sits on the near plane, on the same 3D line (right = 1, up = -1 all along).
        self.assertAlmostEqual(p0[0], self.W / 2 + self.F * 1.0 / E.NEAR_PLANE, places=9)
        self.assertAlmostEqual(p0[1], self.H / 2 + self.F * 1.0 / E.NEAR_PLANE, places=9)
        self.assertAlmostEqual(p1[0], self.W / 2 + self.F * 1.0 / 8.0, places=9)

    def test_box_behind_camera_is_skipped(self):
        cam = np.array([[-3.0, 0.0, 0.0], [-1.0, 1.0, 1.0]])
        self.assertEqual(E.project_edges(cam, self.F, self.W, self.H, edges=((0, 1),)), [])


def _cube_edges():
    idx = {(i, j, k): 4 * i + 2 * j + k for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    out = []
    for (i, j, k), a in idx.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            key = (i + di, j + dj, k + dk)
            if key in idx:
                out.append((a, idx[key]))
    return tuple(out)


class TestFootprint(unittest.TestCase):
    def test_corners_follow_heading(self):
        corners = E._footprint(np.array([0.0, 0.0, math.pi / 2]), (1.0, 2.0, 0.5))
        # Centre 1 m ahead along +y; long axis along y.
        np.testing.assert_allclose(np.mean(corners, axis=0), [0.0, 1.0], atol=1e-12)
        ys = sorted(c[1] for c in corners)
        self.assertAlmostEqual(ys[-1] - ys[0], 4.0, places=12)


class TestChiSquare(unittest.TestCase):
    def test_known_quantiles(self):
        self.assertAlmostEqual(E.chi2_ppf(0.025, 3), 0.215795, places=5)
        self.assertAlmostEqual(E.chi2_ppf(0.975, 3), 9.348404, places=5)
        # 2 dof is exponential: F(x) = 1 - exp(-x/2)
        self.assertAlmostEqual(E.chi2_ppf(0.025, 2), -2 * math.log(0.975), places=9)
        self.assertAlmostEqual(E.chi2_ppf(0.975, 2), -2 * math.log(0.025), places=9)
        self.assertAlmostEqual(E.chi2_cdf(4.0, 2), E.P_INSIDE_2SIGMA_2D, places=12)

    def test_wilson_hilferty_switch(self):
        for n in (101, 150, 400):
            dof = 3 * n
            wh = E.mean_nees_band(n)
            exact = (E.chi2_ppf(0.025, dof) / n, E.chi2_ppf(0.975, dof) / n)
            np.testing.assert_allclose(wh, exact, rtol=1e-3)


class TestFrame(unittest.TestCase):
    def test_heading_maps_to_same_world_direction(self):
        # CARLA yaw 90 deg points along CARLA +y. In the filter frame that is -y.
        s = E.to_filter(0.0, 0.0, math.radians(90.0))
        direction = (math.cos(s[2]), math.sin(s[2]))
        np.testing.assert_allclose(direction, (0.0, -1.0), atol=1e-12)
        np.testing.assert_allclose(E.to_carla_xy(E.to_filter(3.0, 4.0, 0.0)[:2]), (3.0, 4.0))


class TestConsistency(unittest.TestCase):
    """The mock is generated by the filter's own models, so a correct filter must be consistent."""

    def test_mock_run_is_consistent(self):
        runs = []
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(5):
                cfg = E.Cfg(steps=1000, seed=seed)
                runs.append(E.run(E.MockWorld(cfg), cfg, None, log_root=Path(tmp)))
        nees_mean = np.mean([s.nees_mean for s in runs])
        nis_norm = np.mean([s.nis_norm_mean for s in runs])
        frac_in = np.mean([s.frac_nees_in for s in runs])
        frac_inside = np.mean([s.frac_inside for s in runs])
        self.assertGreater(nees_mean, 2.6)
        self.assertLess(nees_mean, 3.4)
        self.assertGreater(nis_norm, 0.9)
        self.assertLess(nis_norm, 1.1)
        self.assertGreater(frac_in, 0.92)
        self.assertLess(frac_in, 0.98)
        self.assertAlmostEqual(frac_inside, E.P_INSIDE_2SIGMA_2D, delta=0.04)


if __name__ == "__main__":
    unittest.main()

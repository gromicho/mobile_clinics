"""Independent numerical and routing checks, including the stochastic benchmark."""
from dataclasses import replace
import itertools
import os
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import clinic_routing as routing
from clinic_routing import Settings, expected_service, benchmark_segments, history, weights, solve


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def instance(self, seed=0):
        rng = np.random.default_rng(seed)
        xy = rng.random((6, 2)) * 40
        d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        return d, rng.uniform(50, 450, 6), rng.random((6, 3))

    def test_expected_service_against_integration(self):
        for mu in [0, 10, 150, 250, 600]:
            integrated = quad(lambda t: min(250, mu * np.exp(-.15**2/2 + .15*t)) * norm.pdf(t),
                              -10, 10, epsabs=1e-7, points=[] if mu == 0 else [(np.log(250/mu)+.15**2/2)/.15])[0]
            self.assertAlmostEqual(float(expected_service(mu)), integrated, places=5)
        np.testing.assert_equal(expected_service([0, 100, 300], sigma=0), [0, 100, 250])

    def test_secants_are_lower_bounds_with_claimed_error(self):
        cfg = Settings()
        exposure = np.linspace(0, 1, 2001)
        for b in [30, 250, 729]:
            for gamma in [-.8, -.5, 0, .25, .8]:
                slope, intercept, bound = benchmark_segments(b, gamma, cfg)
                approximation = (slope[:, None]*exposure + intercept[:, None]).min(axis=0)
                exact = expected_service(b*(1+gamma*exposure))
                self.assertGreaterEqual(float((exact-approximation).min()), -1e-9)
                self.assertLessEqual(float((exact-approximation).max()), bound+1e-8)

    def test_benchmark_bounds_exhaustive_optimum(self):
        for seed, gamma in [(0, -.5), (1, .25), (2, .5), (3, -.8)]:
            d, base, feat = self.instance(seed)
            cfg = Settings(stops=4, benchmark_error=.1)
            result = solve(d, base, feat, 0, gamma, cfg, "benchmark", env=self.env)
            best = -float("inf")
            for tail_size in range(cfg.stops):
                for tail in itertools.permutations(range(1, 6), tail_size):
                    route = [0, *tail]
                    y = np.zeros(6); y[route] = 1
                    served = expected_service(base*(1+gamma*np.minimum(1, weights(d, cfg.range_km)@y)))
                    value = served[route].sum()-sum(d[a,b] for a,b in zip(route[:-1],route[1:]))
                    best = max(best, value)
            self.assertLessEqual(result["expected_objective"], best+1e-6)
            self.assertGreaterEqual(result["expected_objective_upper_bound"], best-1e-6)
            self.assertLessEqual(best-result["expected_objective"], .101)

    def test_one_stop_and_negative_predictions(self):
        d, base, feat = self.instance()
        model = LinearRegression().fit(np.vstack([np.zeros(4), np.ones(4)]), [-10, -10])
        for variant in ["static", "embedded", "benchmark"]:
            result = solve(d, base, feat, 0, -.5, Settings(stops=1), variant, model, env=self.env)
            self.assertEqual(result["order"], [0])
            self.assertEqual(result["km"], 0)
            if variant != "benchmark": self.assertAlmostEqual(result["predicted_service"], 0)

    def test_history_independent_of_architecture_and_call_order(self):
        d, base, feat = self.instance()
        cfg = Settings(stops=4, periods=20, test_periods=10)
        W = weights(d, cfg.range_km)
        first = history(base, feat, W, -.5, cfg)
        history(base, feat, W, .5, cfg, test=True)
        second = history(base, feat, W, -.5, replace(cfg, hidden=(2,)))
        for a,b in zip(first, second): np.testing.assert_array_equal(a,b)
        self.assertTrue(np.any(first[0][:,3] == 0))
        self.assertFalse(np.array_equal(first[1][:60], history(base,feat,W,-.5,cfg,test=True)[1]))

    def test_invalid_parameters(self):
        d, base, feat = self.instance()
        for cfg in [Settings(stops=0), Settings(stops=7), Settings(range_km=0), Settings(sigma=-1)]:
            with self.assertRaises(ValueError):
                routing.build_model(d, base, feat, 0, -.5, cfg, "benchmark", env=self.env)

    def test_infeasibility_reports_status_without_reading_solution(self):
        d, base, feat = self.instance()
        original = routing.build_model
        def infeasible(*args, **kwargs):
            model, x, y, z, bound = original(*args, **kwargs)
            model.addConstr(y[0] == 0)
            return model, x, y, z, bound
        with patch.object(routing, "build_model", side_effect=infeasible):
            with self.assertRaisesRegex(RuntimeError, "No feasible route"):
                solve(d, base, feat, 0, -.5, Settings(stops=4), "benchmark", env=self.env)

    @unittest.skipUnless(os.environ.get("REQUIRE_RESTRICTED_LICENSE") == "1", "Restricted-licence profile only")
    def test_restricted_license_is_really_active(self):
        with gp.Model(env=self.env) as model:
            model.addVars(2001)
            with self.assertRaises(gp.GurobiError) as caught:
                model.optimize()
            self.assertEqual(caught.exception.errno, 10010)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import random
import sys
import unittest
from collections import Counter
from pathlib import Path


MODULE = Path(__file__).parents[1] / "source/booster_train/booster_train/tasks/manager_based/locomotion/goto/core.py"
spec = importlib.util.spec_from_file_location("goto_core", MODULE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class GoToCoreTests(unittest.TestCase):
    def test_world_to_body(self):
        dx, dy, yaw = core.world_to_body_goal((1.0, 2.0, math.pi / 2), (2.0, 2.0, math.pi))
        self.assertAlmostEqual(dx, 0.0, places=7)
        self.assertAlmostEqual(dy, -1.0, places=7)
        self.assertAlmostEqual(yaw, math.pi / 2, places=7)

    def test_wrap_boundary(self):
        self.assertAlmostEqual(core.wrap_to_pi(math.pi), -math.pi)
        self.assertAlmostEqual(core.wrap_to_pi(-math.pi), -math.pi)
        self.assertTrue(-math.pi <= core.wrap_to_pi(123.4) < math.pi)

    def test_sin_cos_observation_is_continuous_at_pi(self):
        eps = 1e-7
        left = (math.sin(math.pi - eps), math.cos(math.pi - eps))
        right = (math.sin(-math.pi + eps), math.cos(-math.pi + eps))
        self.assertLess(math.dist(left, right), 3 * eps)

    def test_reward_increases_near_goal(self):
        far = math.exp(-0.2 * core.constellation_distance(1.0, 0.0, 0.0))
        near = math.exp(-0.2 * core.constellation_distance(0.1, 0.0, 0.0))
        self.assertGreater(near, far)

    def test_radius_increases_orientation_sensitivity(self):
        small = core.constellation_distance(0.0, 0.0, 0.5, inertia=0.25)
        large = core.constellation_distance(0.0, 0.0, 0.5, inertia=4.0)
        self.assertGreater(large, small)

    def test_explicit_matches_analytic(self):
        for radius in (0.2, 1.0, 2.0):
            for angle in (-math.pi, -0.7, 0.0, 1.4, math.pi):
                analytic = core.constellation_distance(0.4, -0.2, angle, inertia=radius**2)
                explicit = core.explicit_constellation_distance(0.4, -0.2, angle, radius)
                self.assertAlmostEqual(analytic, explicit, places=12)

    def test_sampler_range_and_probabilities(self):
        rng = random.Random(7)
        cfg = core.GoalSamplingConfig()
        count = Counter()
        for _ in range(100_000):
            x, y, yaw, category = core.sample_relative_goal(rng, cfg)
            self.assertLessEqual(abs(x), 2.0)
            self.assertLessEqual(abs(y), 1.5)
            self.assertTrue(-math.pi <= yaw <= math.pi)
            count[category] += 1
        for category, expected in enumerate(cfg.probabilities):
            self.assertAlmostEqual(count[category] / 100_000, expected, delta=0.006)


if __name__ == "__main__":
    unittest.main()


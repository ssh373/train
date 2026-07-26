import importlib.util
import math
import random
import sys
import unittest
import ast
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
        far = core.constellation_reward(1.0, 0.0, 0.0)
        near = core.constellation_reward(0.1, 0.0, 0.0)
        self.assertGreater(near, far)

    def test_radius_increases_orientation_sensitivity(self):
        small = core.constellation_distance(0.0, 0.0, 0.5, radius=0.5)
        large = core.constellation_distance(0.0, 0.0, 0.5, radius=2.0)
        self.assertGreater(large, small)

    def test_explicit_matches_analytic(self):
        for radius in (0.2, 1.0, 2.0):
            for angle in (-math.pi, -0.7, 0.0, 1.4, math.pi):
                analytic = core.constellation_distance(0.4, -0.2, angle, radius=radius)
                explicit = core.explicit_constellation_distance(0.4, -0.2, angle, radius)
                self.assertAlmostEqual(analytic, explicit, places=12)

    def test_paper_reward_primitives(self):
        self.assertEqual(core.constellation_reward(0.0, 0.0, 0.0), 1.0)
        self.assertEqual(core.base_height_reward(0.58, 0.58), 1.0)
        self.assertAlmostEqual(core.base_height_reward(0.63, 0.58), math.exp(-1.0))
        self.assertEqual(core.action_difference_reward((0.1, -0.2), (0.1, -0.2)), 1.0)
        self.assertEqual(core.normalized_torque_reward((0.0, 0.0)), 1.0)
        self.assertAlmostEqual(core.normalized_torque_reward((1.0, 1.0)), math.exp(-0.02))

    def test_airtime_semantics(self):
        for air_time, expected in ((0.2, -0.2), (0.4, 0.0), (0.6, 0.2)):
            self.assertAlmostEqual(core.feet_airtime_reward(False, (air_time, 0.0), (True, False)), expected)
        self.assertEqual(core.feet_airtime_reward(False, (0.6, 0.6), (False, False)), 0.0)
        self.assertEqual(core.feet_airtime_reward(True, (0.0, 0.0), (False, False)), 1.0)

    def test_joint_limits_and_tilt_dead_zone(self):
        self.assertEqual(core.joint_limit_cost((0.0,), (-1.0,), (1.0,)), 0.0)
        self.assertGreater(core.joint_limit_cost((1.1,), (-1.0,), (1.0,)), 0.0)
        self.assertEqual(core.tilt_safety_cost(math.radians(20.0)), 0.0)
        self.assertGreater(core.tilt_safety_cost(math.radians(40.0)), 0.0)
        self.assertEqual(core.tilt_safety_cost(math.radians(60.0)), 1.0)

    def test_strict_config_has_exact_reward_set_and_no_push(self):
        env_cfg = MODULE.parents[1] / "robots/k1/goto/env_cfg.py"
        tree = ast.parse(env_cfg.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        rewards = {
            node.targets[0].id for node in classes["RewardsCfg"].body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(rewards, {
            "constellation", "base_height", "feet_airtime", "feet_orientation", "feet_position",
            "base_acceleration", "action_difference", "normalized_torque", "joint_limits",
            "tilt_safety", "undesired_contact",
        })
        source = env_cfg.read_text(encoding="utf-8")
        self.assertIn("self.events.recovery_push = None", source)
        self.assertIn("self.events.sustained_push = None", source)
        self.assertIn("self.events.recovery_push = robust_events.recovery_push", source)
        self.assertIn("self.events.sustained_push = robust_events.sustained_push", source)

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

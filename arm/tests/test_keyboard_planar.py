from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arm_backend import load_sdk
load_sdk()
"""Offline position/heading, Home and inherited hardware guard regressions."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import arm_config as cfg
from core.state_machine import LIMITS
from core.kinematics import PlanarKinematics


class PlanarTests(unittest.TestCase):
    def setUp(self):
        self.k = PlanarKinematics()

    def test_local_steps_follow_heading_and_keep_other_joints(self):
        for heading in (-90000, 0, 45000, 90000):
            q = [heading, 60000, -60000, 12000, 20000, 6000]
            for axis, direction in (('dx', 1), ('dx', -1), ('dz', 1), ('dz', -1)):
                with self.subTest(heading=heading, axis=axis, direction=direction):
                    r = self.k.step(q, **{axis: direction})
                    self.assertIsNotNone(r)
                    for i in (0, 3, 4, 5):
                        self.assertEqual(r[i], q[i])
                    yaw = math.radians(heading/1000)
                    expected = (np.array([math.cos(yaw), math.sin(yaw), 0])
                                if axis == 'dx' else np.array([0, 0, 1])) * direction
                    np.testing.assert_allclose(self.k.position(r)-self.k.position(q), expected, atol=0.03)
                    for value, (lo, hi), seed in zip(r, LIMITS, q):
                        self.assertTrue(lo <= value <= hi)
                        self.assertLessEqual(abs(value-seed), 1000)

    def test_home_can_move_forward_and_up_but_rejects_out_of_range(self):
        self.assertIsNotNone(self.k.step([0]*6, dx=1))
        self.assertIsNotNone(self.k.step([0]*6, dz=1))
        self.assertIsNone(self.k.step([0]*6, dx=-1))
        self.assertIsNone(self.k.step([0]*6, dz=-1))

    def test_invalid_and_singular_requests_rejected(self):
        self.assertIsNone(self.k.step([0]*6, dx=float('nan')))
        self.assertIsNone(self.k.step([0]*6, dx=100))
        self.assertIsNone(self.k.step([0, -2000, 0, 0, 0, 0], dx=1))
        with patch.object(self.k, 'local_position', return_value=np.zeros(3)):
            self.assertIsNone(self.k.step([0]*6, dx=1))


if __name__ == "__main__":unittest.main()

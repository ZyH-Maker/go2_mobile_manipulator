from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arm_backend import load_sdk
load_sdk()
"""Configuration errors must be rejected before the CAN interface opens."""
import unittest
from unittest.mock import patch

import arm_config as cfg
from core.kinematics import PlanarKinematics


class ConfigTests(unittest.TestCase):
    def test_default_configuration_valid(self):
        cfg.validate()

    def test_invalid_numbers_rejected_before_connecting(self):
        cases = [('CONTROL_SPEED_PERCENT', 101), ('RECOVERY_SPEED_PERCENT', 2.5),
                 ('KEY_STEP_SECONDS', 0), ('HOME_SPEED_DEG_S', float('nan')),
                 ('TRACKING_LIMIT_DEG', float('inf')), ('FEEDBACK_TIMEOUT_S', -1),
                 ('IK_MAX_ITERATIONS', 1.5), ('HOME_CONTINUE_AFTER_FINISH', 1)]
        for name, value in cases:
            with self.subTest(name=name), patch.object(cfg, name, value):
                with self.assertRaisesRegex(ValueError, 'config.py'):
                    cfg.validate()

    def test_bad_limits_and_inconsistent_timeouts_rejected(self):
        for name, value in [('JOINT_LIMITS_DEG', None), ('JOINT_LIMITS_DEG', [1]*6),
                            ('JOINT_LIMITS_DEG', [(1, 2)]*6), ('RESET_TIMEOUT_S', 0.5),
                            ('ENABLE_TIMEOUT_S', 0.1), ('IK_SUBSTEP_MM', 3.0)]:
            with self.subTest(name=name), patch.object(cfg, name, value):
                with self.assertRaisesRegex(ValueError, 'config.py'):
                    cfg.validate()

    def test_finite_difference_setting_keeps_derivative_units(self):
        q = [45000, 60000, -60000, 12000, 20000, 6000]
        solver = PlanarKinematics()
        with patch.object(cfg, 'IK_DIFFERENCE_STEP_DEG', 0.002):
            result = solver.step(q, dx=1)
        self.assertIsNotNone(result)
        error = solver.local_position(result)-solver.local_position(q)-[1, 0, 0]
        self.assertLess(float((error @ error)**0.5), cfg.IK_ACCEPTANCE_MM)

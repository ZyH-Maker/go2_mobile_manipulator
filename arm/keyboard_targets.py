"""Gamepad-style key-to-joint-target calculation; no ROS or CAN connection."""
import math
import numpy as np
from arm_common import PROJECT_ROOT
import arm_config as cfg
from arm_trajectory import MAX_SPEED_RAD_S

from core.keyboard_motion import MOTION_KEYS, step_target


class KeyboardTargets:
    def __init__(self, kinematics=None):
        self.kinematics=kinematics
        self.last_character=-math.inf

    def step(self, key, start, actual, now):
        if key not in MOTION_KEYS or now-self.last_character < cfg.KEY_STEP_SECONDS-1e-9:
            return None
        self.last_character=now
        # Match Gamepad's nearby-solution IK units (0.001 degree).
        q=np.rint(np.rad2deg(start)*1000).astype(int).tolist()
        actual=np.asarray(actual)
        if np.max(np.abs(np.asarray(start)-actual)) > math.radians(cfg.TRACKING_LIMIT_DEG):
            return None
        if key in 'qews' and self.kinematics is None:
            from arm_backend import load_sdk
            load_sdk()
            from core.kinematics import PlanarKinematics
            self.kinematics=PlanarKinematics()
        q=step_target(key,q,self.kinematics)
        if q is None:return None
        candidate=np.deg2rad(np.asarray(q)/1000.)
        if np.max(np.abs(candidate-actual)) > math.radians(cfg.TRACKING_LIMIT_DEG):return None
        duration=max(cfg.KEY_STEP_SECONDS,float(np.max(np.abs(candidate-np.asarray(start))))/MAX_SPEED_RAD_S)
        return candidate,duration


class JointKeyboardTargets:
    """One selected joint, same trajectory interface as Cartesian targets."""
    def __init__(self):
        self.selected = 0
        self.step_rad = math.radians(cfg.JOINT_DEFAULT_STEP_DEG)
        self.last_character = -math.inf

    def step(self, key, start, actual, now):
        if key not in ('j', 'l') or now-self.last_character < cfg.JOINT_KEY_INTERVAL_S-1e-9:
            return None
        self.last_character = now
        start = np.asarray(start)
        actual = np.asarray(actual)
        if np.max(np.abs(start-actual)) > math.radians(cfg.TRACKING_LIMIT_DEG):
            return None
        candidate = start.copy()
        lo, hi = cfg.JOINT_LIMITS_DEG[self.selected]
        candidate[self.selected] = np.clip(start[self.selected] + (-1 if key == 'j' else 1)*self.step_rad,
                                           math.radians(lo), math.radians(hi))
        if np.array_equal(candidate, start):
            return None
        if np.max(np.abs(candidate-actual)) > math.radians(cfg.TRACKING_LIMIT_DEG):
            return None
        duration = max(cfg.JOINT_KEY_INTERVAL_S, float(np.max(np.abs(candidate-start)))/MAX_SPEED_RAD_S)
        return candidate, duration

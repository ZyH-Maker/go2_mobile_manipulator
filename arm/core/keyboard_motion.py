"""Shared character-to-position step for the ROS keyboard target node.

Inputs and outputs are six joint angles in 0.001 degrees. This module does
not connect to CAN or import ROS/the SDK; planar IK is supplied by the caller.
"""
import math
import time
import arm_config as cfg

MOTION_KEYS = frozenset('adqewsjluo')


def step_target(key, seed, kinematics=None):
    if key not in MOTION_KEYS:
        return None
    candidate = list(seed)
    dt = cfg.KEY_STEP_SECONDS
    amount = cfg.JOINT_SPEED_DEG_S * cfg.SPEED_FACTOR * dt * 1000
    for joint, positive, negative in ((0, 'a', 'd'), (4, 'l', 'j'), (5, 'o', 'u')):
        candidate[joint] += round(amount * ((key == positive) - (key == negative)))
    candidate = [max(cfg.mdeg(lo), min(cfg.mdeg(hi), value))
                 for value, (lo, hi) in zip(candidate, cfg.JOINT_LIMITS_DEG)]
    dx = int(key == 'q') - int(key == 'e')
    dz = int(key == 'w') - int(key == 's')
    if dx or dz:
        if kinematics is None:
            raise ValueError('位置移动需要平面逆解器')
        step = cfg.LINEAR_SPEED_MM_S * cfg.SPEED_FACTOR * dt
        count = max(1, math.ceil(step / cfg.IK_SUBSTEP_MM))
        deadline = time.perf_counter() + cfg.IK_FRAME_TIMEOUT_S
        for _ in range(count):
            candidate = kinematics.step(candidate, dx=dx * step / count,
                                         dz=dz * step / count)
            if candidate is None or time.perf_counter() > deadline:
                return None
    return candidate

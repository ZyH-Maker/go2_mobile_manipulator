"""Small local X/Z IK steps using the installed PiPER SDK's offset DH model.

Positions are the joint-6 flange origin in mm, not the gripper fingertips.
Only J2/J3 are solved; J1/J4/J5/J6 remain exactly at their seed values.
"""
import math
import time

import arm_config as cfg

import numpy as np
from piper_sdk import C_PiperForwardKinematics

from core.state_machine import LIMITS


class PlanarKinematics:
    def __init__(self):
        self.fk = C_PiperForwardKinematics(dh_is_offset=cfg.DH_OFFSET)

    def position(self, joints):
        return np.array(self.fk.CalFK([math.radians(q / 1000) for q in joints])[-1][:3])

    def local_position(self, joints):
        x, y, z = self.position(joints)
        yaw = math.radians(joints[0] / 1000)
        return np.array([math.cos(yaw) * x + math.sin(yaw) * y,
                         -math.sin(yaw) * x + math.cos(yaw) * y, z])

    def step(self, seed, dx=0.0, dz=0.0):
        """Return bounded nearby joint targets or None; never switch IK branches."""
        if (len(seed) != 6 or not np.isfinite(seed).all()
                or not math.isfinite(dx) or not math.isfinite(dz)
                or math.hypot(dx, dz) > cfg.IK_MAX_REQUEST_MM):
            return None
        seed = np.array(seed, dtype=float)
        if any(not lo <= q <= hi for q, (lo, hi) in zip(seed, LIMITS)):
            return None
        desired = self.local_position(seed) + [dx, 0, dz]
        q = seed.copy()
        # Bound work so a failed solve cannot starve CAN reception / stop keys.
        deadline = time.perf_counter() + cfg.IK_SOLVE_TIMEOUT_S
        lower = np.maximum([LIMITS[1][0], LIMITS[2][0]], seed[1:3] - cfg.mdeg(cfg.IK_MAX_JOINT_CHANGE_DEG))
        upper = np.minimum([LIMITS[1][1], LIMITS[2][1]], seed[1:3] + cfg.mdeg(cfg.IK_MAX_JOINT_CHANGE_DEG))
        for _ in range(cfg.IK_MAX_ITERATIONS):
            if time.perf_counter() > deadline:
                return None
            pos = self.local_position(q)
            error = (desired - pos)[[0, 2]]
            if np.linalg.norm(error) < cfg.IK_CONVERGENCE_MM:
                result = np.rint(q).astype(int).tolist()
                if np.linalg.norm(self.local_position(result) - desired) <= cfg.IK_ACCEPTANCE_MM:
                    return result
                return None
            columns = []
            for j in (1, 2):
                probe = q.copy()
                probe[j] += cfg.IK_DIFFERENCE_STEP_DEG * 1000
                columns.append((self.local_position(probe) - pos)[[0, 2]] / (cfg.IK_DIFFERENCE_STEP_DEG * 1000))
            jac = np.column_stack(columns)
            # Reject near-singular poses instead of taking a large joint jump.
            if np.linalg.svd(jac, compute_uv=False)[-1] < cfg.IK_MIN_SINGULAR_VALUE:
                return None
            delta = np.linalg.solve(jac.T @ jac + np.eye(2) * cfg.IK_DAMPING, jac.T @ error)
            if not np.isfinite(delta).all():
                return None
            q[1:3] = np.clip(q[1:3] + np.clip(delta, -cfg.mdeg(cfg.IK_ITERATION_STEP_DEG), cfg.mdeg(cfg.IK_ITERATION_STEP_DEG)), lower, upper)
        return None

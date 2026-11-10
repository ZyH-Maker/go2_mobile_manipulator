"""Hardware-free helpers for the Go2 policy deployment."""
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np

ARM_NAMES = ['joint%d' % i for i in range(1, 7)]
LEG_NAMES = ['%s_%s_joint' % (leg, joint) for leg in ['FL', 'FR', 'RL', 'RR']
             for joint in ['hip', 'thigh', 'calf']]
L1, Y, B, L2, R2 = 2, 2048, 512, 32, 16


def arm_limits_from_urdf(path):
    """Read hardware-profile limits in feedback order, independent of RL geometry."""
    root = ET.parse(path).getroot()
    limits = []
    for name in ARM_NAMES:
        joints = root.findall("joint[@name='%s']" % name)
        if len(joints) != 1 or joints[0].find('limit') is None:
            raise ValueError('机械臂 URDF 缺少或重复关节限位：'+name)
        limit = joints[0].find('limit')
        limits.append([float(limit.attrib['lower']), float(limit.attrib['upper'])])
    limits = np.asarray(limits)
    if not np.isfinite(limits).all() or (limits[:, 0] >= limits[:, 1]).any():
        raise ValueError('机械臂 URDF 关节限位无效')
    return limits


def vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError('需要长度为 %d 的有限数值向量' % size)
    return value


def named_arm(names, positions, velocities):
    if len(set(names)) != len(names):
        raise ValueError('关节名称重复')
    ids = [list(names).index(n) for n in ARM_NAMES]
    return vector([positions[i] for i in ids], 6), vector([velocities[i] for i in ids], 6)


def remote(payload):
    payload = bytes(payload)
    if len(payload) != 40 or payload[:2] != b'\x55\x51':
        raise ValueError('宇树遥控器数据包无效')
    keys = struct.unpack_from('<H', payload, 2)[0]
    lx, rx, ry, spare, ly = struct.unpack_from('<5f', payload, 4)
    axes = vector([lx, ly, rx], 3)
    if np.max(np.abs(axes)) > 1.01:
        raise ValueError('遥控器摇杆值超出范围')
    return keys, axes


def neutral(axes):
    return np.max(np.abs(axes)) < 0.08


class Lifecycle:
    """Go2 release, standing, policy and latched stop states."""
    def __init__(self):
        self.phase = 'WAIT'
        self.reason = ''
        self.paused = False

    def action(self, event, sticks_neutral=True):
        if event == 'stop':
            self.reason = '操作者停止'
            self.phase = 'FAULT'
        elif event == 'prepare' and self.phase == 'WAIT' and sticks_neutral:
            self.phase = 'RELEASE'
        elif event == 'lowlevel_confirmed' and self.phase == 'RELEASE':
            self.phase = 'STAND'
        elif event == 'stand_ready' and self.phase == 'STAND':
            self.phase = 'READY'
        elif event == 'rl' and self.phase == 'READY' and sticks_neutral:
            self.phase = 'POLICY'
        elif event == 'pause' and self.phase == 'POLICY':
            self.paused = True
        elif event == 'rl' and self.phase == 'POLICY' and self.paused and sticks_neutral:
            self.paused = False

    def fault(self, reason):
        self.reason = str(reason)
        self.phase = 'FAULT'


def check_feedback(now, receipts, timeout=0.1):
    if any(not math.isfinite(t) or now-t > timeout or now < t for t in receipts):
        raise ValueError('反馈超时')


def clamp_leg_target(target, q, dq, kp, kd, torque, limits=None):
    target, q, dq = (vector(v, 12) for v in (target, q, dq))
    if limits is None:
        # Original deployment's torque saturation; no extra model-range gate.
        return np.clip(target, q+(-torque+kd*dq)/kp, q+(torque+kd*dq)/kp)
    target = np.clip(target, limits[:, 0], limits[:, 1])
    lower = np.maximum(limits[:,0], q+(-torque+kd*dq)/kp)
    upper = np.minimum(limits[:,1], q+(torque+kd*dq)/kp)
    if (lower > upper).any():
        raise ValueError('无法同时满足关节位置及 PD 力矩限制')
    return np.clip(target, lower, upper)


def validate_policy_pd(kp, kd):
    """Match HAL gain bounds; positive Kp is needed by torque saturation."""
    if not math.isfinite(kp) or not 0 < kp <= 70:
        raise ValueError('策略 Kp 必须为有限数值，且满足 0 < Kp <= 70')
    if not math.isfinite(kd) or not 0 <= kd <= 6:
        raise ValueError('策略 Kd 必须为有限数值，且满足 0 <= Kd <= 6')
    return float(kp), float(kd)

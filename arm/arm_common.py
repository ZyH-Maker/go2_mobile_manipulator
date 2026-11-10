"""ROS arm names and feedback parsing, independent of Go2/LocoPilot."""
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARM_NAMES = ['joint%d' % i for i in range(1, 7)]


def named_arm(names, positions, velocities):
    if len(set(names)) != len(names):
        raise ValueError('关节名称重复')
    ids = [list(names).index(n) for n in ARM_NAMES]
    q = np.asarray([positions[i] for i in ids], dtype=float)
    dq = np.asarray([velocities[i] for i in ids], dtype=float)
    if q.shape != (6,) or dq.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('机械臂反馈需要六轴有限数值')
    return q, dq

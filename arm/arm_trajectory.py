"""Position-only JointTrajectory validation and piecewise-linear interpolation."""
import math
import numpy as np
from arm_common import ARM_NAMES
import arm_config as cfg

LIMITS_RAD = np.deg2rad(np.asarray(cfg.JOINT_LIMITS_DEG, dtype=float))
MAX_SPEED_RAD_S = math.radians(25.)


def duration_seconds(duration):
    if duration.sec < 0 or not 0 <= duration.nanosec < 1000000000:
        raise ValueError('轨迹时间无效')
    return duration.sec + duration.nanosec/1e9


class JointTrajectoryPlan:
    def __init__(self, message, start_q, start_time):
        if len(message.joint_names) != 6 or set(message.joint_names) != set(ARM_NAMES):
            raise ValueError('轨迹必须完整包含 joint1～joint6，各一次')
        order = [message.joint_names.index(n) for n in ARM_NAMES]
        previous_q = np.asarray(start_q, dtype=float)
        if previous_q.shape != (6,) or not np.isfinite(previous_q).all():
            raise ValueError('轨迹初始状态无效')
        if not message.points:
            raise ValueError('轨迹为空；取消请调用 hold 服务')
        self.times, self.positions = [0.], [previous_q.copy()]
        previous_t = 0.
        for i, point in enumerate(message.points):
            if len(point.positions) != 6:
                raise ValueError('轨迹点需要六个关节角')
            if point.velocities or point.accelerations or point.effort:
                raise ValueError('当前执行接口仅支持位置轨迹，请留空速度、加速度和力矩数组')
            q = np.asarray(point.positions, dtype=float)[order]
            if not np.isfinite(q).all() or (q < LIMITS_RAD[:,0]-1e-9).any() or (q > LIMITS_RAD[:,1]+1e-9).any():
                raise ValueError('轨迹目标非有限数或超出标准 Piper 范围')
            t = duration_seconds(point.time_from_start)
            if t < previous_t or (i > 0 and t == previous_t):
                raise ValueError('轨迹时间必须严格递增')
            dt = t - previous_t
            if dt == 0:
                if np.max(np.abs(q-previous_q)) > 1e-6:
                    raise ValueError('零时刻目标必须与当前目标一致，不能瞬间跳变')
            elif np.max(np.abs(q-previous_q))/dt > MAX_SPEED_RAD_S + 1e-6:
                raise ValueError('轨迹关节目标变化率超过 25°/s，请延长执行时间')
            self.times.append(t);self.positions.append(q)
            previous_q, previous_t = q, t
        self.start_time = float(start_time)
        self.end_time = self.start_time + self.times[-1]

    def sample(self, now):
        elapsed = now - self.start_time
        if elapsed <= 0:
            return self.positions[0].copy()
        for i in range(1, len(self.times)):
            if elapsed < self.times[i]:
                ratio = (elapsed-self.times[i-1])/(self.times[i]-self.times[i-1])
                return self.positions[i-1]*(1-ratio)+self.positions[i]*ratio
        return self.positions[-1].copy()

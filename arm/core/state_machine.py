"""PiPER feedback and position takeover state machine, used by the executor.

No keyboard loop, CAN connection or SDK initialization occurs in this module.
The robot instance is injected by arm_backend.py.
"""
import math
import time
import arm_config as cfg

LIMITS = tuple(tuple(cfg.mdeg(x) for x in pair) for pair in cfg.JOINT_LIMITS_DEG)
FEEDBACK_IDS = (0x2A1, 0x2A5, 0x2A6, 0x2A7, *range(0x261, 0x267))
FAULT_FLAGS = ('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
               'driver_overheating', 'collision_status', 'driver_error_status',
               'stall_status')


class Feedback:
    def __init__(self, robot):
        self.robot = robot
        self.last_seen = {}

    def receive(self, message):
        if (message.is_error_frame or message.is_remote_frame
                or message.is_extended_id or len(message.data) != 8
                or message.arbitration_id not in FEEDBACK_IDS):
            return
        self.robot.ParseCANFrame(message)
        # Use the kernel receive time so queued old packets cannot appear fresh.
        age = max(0.0, time.time() - message.timestamp)
        self.last_seen[message.arbitration_id] = time.monotonic() - age

    def missing(self):
        now = time.monotonic()
        return [can_id for can_id in FEEDBACK_IDS
                if now - self.last_seen.get(can_id, -math.inf) > cfg.FEEDBACK_TIMEOUT_S]

    def joints(self):
        state = self.robot.GetArmJointMsgs().joint_state
        return [getattr(state, f'joint_{i}') for i in range(1, 7)]

    def problem(self, allow_stopped=False, check_limits=True):
        missing = self.missing()
        if missing:
            return f'反馈缺失或超过 {cfg.FEEDBACK_TIMEOUT_S:g} 秒: ' + ', '.join(hex(x) for x in missing)
        state = self.robot.GetArmStatus().arm_status
        if state.err_code:
            return f'机械臂故障码: {state.err_code}'
        allowed = (0, 1) if allow_stopped else (0,)
        if state.arm_status not in allowed:
            return f'机械臂状态异常: {state.arm_status}'
        motor_problem = self.motor_problem()
        if motor_problem:
            return motor_problem
        if check_limits:
            return self.range_problem()
        return ''

    def motor_problem(self):
        motors = self.robot.GetArmLowSpdInfoMsgs()
        for i in range(1, 7):
            status = getattr(motors, f'motor_{i}').foc_status
            for flag in FAULT_FLAGS:
                if getattr(status, flag):
                    return f'J{i}: {flag}'
        return ''

    def range_problem(self):
        # The target is always bounded; allow small feedback/encoder offsets.
        for i, (value, (lower, upper)) in enumerate(zip(self.joints(), LIMITS)):
            if not lower - cfg.mdeg(cfg.RANGE_TOLERANCE_DEG) <= value <= upper + cfg.mdeg(cfg.RANGE_TOLERANCE_DEG):
                return f'J{i + 1} 当前 {value / 1000:.3f}° 超出范围；R 低速回最近边界'
        return ''


class PositionController:
    def __init__(self, robot, feedback):
        self.robot = robot
        self.feedback = feedback
        self.selected = 0
        self.step = cfg.mdeg(cfg.JOINT_DEFAULT_STEP_DEG)
        self.speed = cfg.JOINT_CONTROL_SPEED_PERCENT
        self.target = None
        self.phase = 'monitor'
        self.owns_control = False
        self.enable_deadline = 0.0
        self.last_enable = -math.inf
        self.enable_count = 0
        self.reset_deadline = 0.0
        self.settle_not_before = 0.0
        self.stable_since = None
        self.stable_joints = None
        self.mode_deadline = 0.0
        self.last_send = -math.inf
        self.recover_limits = False
        self.tracking_limits = [cfg.mdeg(cfg.TRACKING_LIMIT_DEG)] * 6
        self.recovery_deadline = 0.0
        self.last_jog = -math.inf
        self.notice = '只读监视；确认运动空间后按 E 接管当前位置。'

    @staticmethod
    def bounded(joints):
        return [max(lo, min(hi, value)) for value, (lo, hi) in zip(joints, LIMITS)]

    def send_target(self):
        self.robot.ModeCtrl(0x01, 0x01, self.speed, 0x00)
        self.robot.JointCtrl(*self.target)
        self.last_send = time.monotonic()

    def stop(self, reason):
        self.phase = 'stopped'
        self.target = None
        self.notice = reason
        # Fast stop is reserved for explicit emergency stop and faults.
        self.robot.MotionCtrl_1(0x01, 0, 0)

    def hold(self, reason='已暂停，保持当前位置；按 E 继续。'):
        if not self.owns_control or self.phase in ('monitor', 'stopped'):
            return
        state = self.robot.GetArmStatus().arm_status
        problem = self.feedback.problem()
        if (problem or state.ctrl_mode != 1 or state.mode_feed != 1
                or not all(self.robot.GetArmEnableStatus())
                or self.phase not in ('active', 'held', 'recovering')):
            self.stop('无法确认当前位置保持条件；已发送急停。' + problem)
            return
        self.target = self.bounded(self.feedback.joints())
        self.speed = cfg.JOINT_CONTROL_SPEED_PERCENT
        self.tracking_limits = [cfg.mdeg(cfg.TRACKING_LIMIT_DEG)] * 6
        self.phase = 'held'
        try:
            self.send_target()
        except Exception:
            self.stop('当前位置保持指令失败；已尝试发送急停。')
            raise
        self.notice = reason

    def take_control(self, recover_limits=False):
        if self.phase in ('reset_pending', 'resetting', 'enabling', 'activating', 'recovering'):
            self.notice = '正在接管/恢复，请等待完成，无需重复按 E/R；空格急停。'
            return
        if self.phase == 'active':
            self.notice = '已接管；按空格停止。'
            return
        problem = self.feedback.problem(allow_stopped=True, check_limits=False)
        if problem:
            self.notice = problem
            return
        actual = self.feedback.joints()
        target = self.bounded(actual)
        distance = max(abs(a - t) for a, t in zip(actual, target))
        if distance > cfg.mdeg(cfg.RECOVERY_MAX_DEG):
            self.notice = f'越界超过 {cfg.RECOVERY_MAX_DEG:g}°，拒绝自动恢复；请核对实际姿态和零位。'
            return
        if distance > cfg.mdeg(cfg.RANGE_TOLERANCE_DEG) and not recover_limits:
            self.notice = self.feedback.range_problem() + '（会产生小幅运动）。'
            return
        was_stopped = self.phase == 'stopped'
        self.recover_limits = recover_limits and distance > cfg.mdeg(cfg.RANGE_TOLERANCE_DEG)
        self.speed = cfg.RECOVERY_SPEED_PERCENT if self.recover_limits else cfg.JOINT_CONTROL_SPEED_PERCENT
        self.target = target
        self.owns_control = True
        state = self.robot.GetArmStatus().arm_status
        # SDK stop demo requires reset + repeated enable. A stopped arm may
        # report NORMAL/STANDBY, so arm_status alone cannot identify the latch.
        if state.ctrl_mode not in (0, 1) or state.mode_feed == 4:
            self.robot.MotionCtrl_1(0x01, 0, 0)
            self.phase = 'reset_pending'
            self.reset_deadline = time.monotonic() + cfg.MODE_EXIT_WAIT_S
            self.notice = '正在退出示教/其他控制模式，随后恢复并重新使能。'
        elif was_stopped or state.arm_status == 1 or state.ctrl_mode == 0:
            self.reset()
        else:
            self.begin_enable()

    def reset(self):
        self.robot.MotionCtrl_1(0x02, 0, 0)
        self.phase = 'resetting'
        self.settle_not_before = time.monotonic() + cfg.RESET_MIN_WAIT_S
        self.reset_deadline = time.monotonic() + cfg.RESET_TIMEOUT_S
        self.stable_since = None
        self.stable_joints = None
        self.notice = '已复位；等待驱动重启和关节反馈稳定，不发送运动目标。'

    def begin_enable(self):
        # Resample after reset, rather than moving back to a pre-stop position.
        actual = self.feedback.joints()
        self.target = self.bounded(actual)
        distances = [abs(a - t) for a, t in zip(actual, self.target)]
        if max(distances) > cfg.mdeg(cfg.RECOVERY_MAX_DEG if self.recover_limits else cfg.RANGE_TOLERANCE_DEG):
            self.stop('恢复期间姿态改变，超出允许范围；请检查后再操作。')
            return
        self.tracking_limits = [max(cfg.mdeg(cfg.TRACKING_LIMIT_DEG), d + cfg.mdeg(cfg.RECOVERY_TRACKING_MARGIN_DEG)) for d in distances]
        self.phase = 'enabling'
        self.enable_deadline = time.monotonic() + cfg.ENABLE_TIMEOUT_S
        self.last_enable = -math.inf
        self.enable_count = 0
        self.notice = '正在使能六个电机，完成前不发送运动目标。'

    def begin_motion(self):
        # Motors may settle during power-up. Never use a stale pre-enable pose.
        actual = self.feedback.joints()
        self.target = self.bounded(actual)
        distances = [abs(a - t) for a, t in zip(actual, self.target)]
        if max(distances) > cfg.mdeg(cfg.RECOVERY_MAX_DEG if self.recover_limits else cfg.RANGE_TOLERANCE_DEG):
            self.stop('使能后姿态超出恢复范围；请检查实际姿态。')
            return
        self.tracking_limits = [max(cfg.mdeg(cfg.TRACKING_LIMIT_DEG), d + cfg.mdeg(cfg.RECOVERY_TRACKING_MARGIN_DEG)) for d in distances]
        self.phase = 'activating'
        self.mode_deadline = time.monotonic() + cfg.MODE_TIMEOUT_S
        self.robot.JointCtrl(*self.target)
        self.send_target()
        self.notice = '六个电机已使能，正在确认 CAN 关节模式。'

    def tick(self):
        if self.phase not in ('reset_pending', 'resetting', 'enabling', 'activating', 'recovering', 'active', 'held'):
            return
        now = time.monotonic()
        state = self.robot.GetArmStatus().arm_status
        preparing = self.phase in ('reset_pending', 'resetting', 'enabling')
        problem = self.feedback.problem(allow_stopped=preparing, check_limits=False)
        # S-V1.5-8 briefly reports low-bit communication faults and zero joint
        # feedback while rebooting its motor interface. The error bits can
        # clear before status 5 clears. Wait passively in both cases here;
        # never allow this state in position control, or extend the deadline.
        initializing = (self.phase in ('resetting', 'enabling')
                        and state.arm_status == 5
                        and state.err_code & ~0x3F == 0
                        and not self.feedback.missing()
                        and not self.feedback.motor_problem())
        if initializing:
            self.stable_since = None
            self.stable_joints = None
            deadline = self.reset_deadline if self.phase == 'resetting' else self.enable_deadline
            if now >= deadline:
                self.stop(f'初始化通信故障持续超时: {state.err_code}；已急停。')
            else:
                self.notice = f'驱动初始化中（通信码 {state.err_code}），等待恢复；未发送运动目标。'
            return
        if problem:
            self.stop(problem + '；已发送停止命令。')
            return
        enabled = all(self.robot.GetArmEnableStatus())
        in_joint_mode = state.ctrl_mode == 1 and state.mode_feed == 1
        if self.phase == 'reset_pending':
            if now >= self.reset_deadline:
                self.reset()
            return
        if self.phase == 'resetting':
            if now >= self.reset_deadline:
                self.stop('复位后反馈未稳定，超时急停；请检查硬件。')
                return
            actual = self.feedback.joints()
            motors = self.robot.GetArmLowSpdInfoMsgs()
            valid = (state.arm_status == 0 and state.err_code == 0
                     and all(getattr(motors, f'motor_{i}').vol > 0 for i in range(1, 7)))
            if (not valid or self.stable_joints is None
                    or any(abs(a - b) > cfg.mdeg(cfg.RESET_STABLE_DELTA_DEG) for a, b in zip(actual, self.stable_joints))):
                self.stable_since = now if valid else None
                self.stable_joints = actual
            elif (now >= self.settle_not_before and self.stable_since is not None
                  and now - self.stable_since >= cfg.RESET_STABLE_TIME_S):
                self.begin_enable()
            return
        if self.phase == 'enabling':
            if now >= self.enable_deadline:
                self.stop('接管超时；已发送停止命令，请检查模式/急停状态。')
            elif (self.enable_count >= cfg.ENABLE_MIN_COUNT and now - self.last_enable >= cfg.ENABLE_INTERVAL_S
                  and enabled and state.arm_status == 0):
                self.begin_motion()
            elif now - self.last_enable >= cfg.ENABLE_INTERVAL_S:
                self.robot.EnablePiper()
                self.last_enable = now
                self.enable_count += 1
            return
        if self.phase == 'activating':
            if not enabled:
                self.stop('切换位置模式时电机失能；已急停。')
            elif in_joint_mode and state.arm_status == 0:
                self.phase = 'recovering' if self.recover_limits else 'active'
                self.recovery_deadline = now + cfg.RECOVERY_TIMEOUT_S
                self.notice = (f'正在以 {cfg.RECOVERY_SPEED_PERCENT}% 速度回最近边界，完成后暂停。' if self.recover_limits
                               else '已接管：1–6 选关节，J/L 移动；P 暂停，空格急停。')
            elif now >= self.mode_deadline:
                self.stop('CAN 关节模式切换超时；已急停。')
            elif now - self.last_send >= cfg.TARGET_REFRESH_INTERVAL_S:
                self.send_target()
            return
        if not enabled or not in_joint_mode:
            self.stop('电机失能或控制模式改变；已发送停止命令。')
            return
        errors = [abs(a - t) for a, t in zip(self.feedback.joints(), self.target)]
        if any(error > limit for error, limit in zip(errors, self.tracking_limits)):
            self.stop('目标与反馈偏差过大；已发送停止命令。')
            return
        if self.phase == 'recovering':
            if max(errors) <= cfg.mdeg(cfg.RECOVERY_ARRIVAL_DEG):
                self.recover_limits = False
                self.tracking_limits = [cfg.mdeg(cfg.TRACKING_LIMIT_DEG)] * 6
                self.phase = 'held'
                self.speed = cfg.JOINT_CONTROL_SPEED_PERCENT
                self.notice = '边界恢复完成，已暂停；按 E 进入键盘控制。'
            elif now >= self.recovery_deadline:
                self.stop('边界恢复超时；已急停，请检查实际姿态/零位。')
                return
        # Keep the requested joint pose refreshed, as in the SDK joint demo.
        if now - self.last_send >= cfg.TARGET_REFRESH_INTERVAL_S:
            self.send_target()

    def jog(self, direction):
        self.tick()
        if self.phase != 'active':
            self.notice = '当前不能移动；先查看状态，再按 E 接管。'
            return
        now = time.monotonic()
        if now - self.last_jog < cfg.JOINT_KEY_INTERVAL_S:
            return
        actual = self.feedback.joints()
        if any(abs(a - t) > cfg.mdeg(cfg.TRACKING_LIMIT_DEG) for a, t in zip(actual, self.target)):
            self.stop(f'目标与反馈偏差超过 {cfg.TRACKING_LIMIT_DEG:g}°；已发送停止命令。')
            return
        candidate = self.target.copy()
        i = self.selected
        lower, upper = LIMITS[i]
        requested = candidate[i] + direction * self.step
        bounded = max(lower, min(upper, requested))
        if (bounded - candidate[i]) * direction <= 0:
            self.notice = f'J{i + 1} 已到关节限位，请向相反方向移动。'
            return
        candidate[i] = bounded
        if abs(candidate[i] - actual[i]) > cfg.mdeg(cfg.TRACKING_LIMIT_DEG):
            self.notice = '等待机械臂到达目标，再继续移动。'
            return
        self.target = candidate
        self.send_target()
        self.last_jog = now
        self.notice = f'J{i + 1} 目标 {candidate[i] / 1000:.3f}°'

    def close(self):
        if self.owns_control:
            self.hold('正常退出：已发送当前位置保持指令。')

"""Single CAN position backend for recovery, hold and trajectory execution.

Constructing this object opens no hardware. request() connects for passive
feedback first; advance() drives the copied Gamepad recovery state machine.
"""
import sys
import time
from pathlib import Path

from arm_common import PROJECT_ROOT
import arm_config as cfg


class ArmRequestRejected(RuntimeError):
    """A precondition rejected takeover before any arm control began."""


def load_sdk():
    runtime = Path(__file__).resolve().parents[1]/'third_party/piper_sdk_runtime'
    if not (runtime/'piper_sdk/__init__.py').is_file():
        raise RuntimeError('缺少恢复用 piper_sdk 0.6.2，请按 README 安装独立 SDK')
    if 'piper_sdk' in sys.modules:
        origin = Path(sys.modules['piper_sdk'].__file__).resolve()
        if runtime.resolve() not in origin.parents:
            raise RuntimeError('已加载其他版本 piper_sdk，请重启并使用恢复入口的固定 SDK')
    sys.path.insert(0, str(runtime))
    from piper_sdk import C_PiperInterface_V2
    return C_PiperInterface_V2


class ArmBackend:
    def __init__(self, channel, report=lambda message: None, robot_factory=None):
        self.channel = channel
        self.report = report
        self.robot_factory = robot_factory
        self.robot = self.feedback = self.controller = self.bus = None
        self.requested = False
        self.collecting = False
        self.recover_requested = True
        self.deadline = 0.
        self.last_notice = None
        self.failed = False

    @property
    def ready(self):
        return bool(self.controller and self.controller.phase == 'held' and not self.failed)

    def connect(self):
        if self.robot is not None:
            return
        cfg.validate()
        try:
            factory = self.robot_factory or load_sdk()
            self.robot = factory(can_name=self.channel, dh_is_offset=1)
            self.robot.ConnectPort(piper_init=False, start_thread=False)
            self.bus = self.robot.GetCanBus().recv_bus
            if self.bus is None:
                raise RuntimeError('机械臂执行层无法连接 CAN')
            from core.state_machine import FEEDBACK_IDS, Feedback, PositionController
            self.bus.set_filters([{'can_id': i, 'can_mask': 0x7ff, 'extended': False} for i in FEEDBACK_IDS])
            self.feedback = Feedback(self.robot)
            self.controller = PositionController(self.robot, self.feedback)
        except Exception:
            self.close()
            raise

    def request(self, recover=True):
        if self.requested or self.failed:
            raise RuntimeError('已接管、正在恢复或故障已锁定，请先检查状态')
        self.connect()
        self.requested = True
        self.recover_requested = recover
        self.collecting = True
        self.deadline = time.monotonic() + 3.
        self.report('先读取完整状态；确认前不使能，准备恢复' if recover else '先读取完整状态；确认前不使能，准备当前位置接管')

    def poll(self):
        if self.bus is None:
            return
        for _ in range(256):
            msg = self.bus.recv(timeout=0.)
            if msg is None:
                break
            self.feedback.receive(msg)

    def command(self, radians):
        import math
        if not self.ready:
            raise RuntimeError('机械臂尚未接管到位')
        q = [round(math.degrees(v)*1000) for v in radians]
        if q != self.controller.bounded(q):
            raise ValueError('轨迹目标越界')
        self.controller.target = q
        self.controller.speed = 100
        self.controller.phase = 'held'

    def hold(self):
        if self.failed:
            return
        if self.ready:
            self.controller.hold('已取消轨迹，保持当前位置')
            if self.controller.phase == 'stopped':
                self.failed = True
        elif self.controller and self.controller.owns_control:
            self.stop('接管/恢复已取消；需要重启执行节点')
        else:
            # Cancel a request still collecting feedback, without transmitting.
            self.requested = self.collecting = False

    def advance(self):
        if not self.requested or self.failed:
            return
        try:
            self.poll()
            if self.collecting:
                if self.feedback.missing():
                    if time.monotonic() >= self.deadline:
                        raise RuntimeError('机械臂恢复所需状态反馈不完整；尚未使能')
                    return
                actual = self.feedback.joints()
                target = self.controller.bounded(actual)
                self.report('机械臂恢复目标（度）：当前 %s → %s；最多每轴 5°，恢复速度 5%%' %
                            ([v/1000 for v in actual], [v/1000 for v in target]))
                self.collecting = False
                self.controller.take_control(recover_limits=self.recover_requested)
                if not self.controller.owns_control:
                    raise ArmRequestRejected(self.controller.notice)
            else:
                self.controller.tick()
            if self.controller.phase == 'stopped':
                raise RuntimeError(self.controller.notice)
            # In-range R requests may finish in active, not recovering. Confirm
            # arrival before reporting held; never let L1 race an unfinished move.
            if self.controller.phase == 'active':
                if max(abs(a-b) for a,b in zip(self.feedback.joints(),self.controller.target)) <= cfg.mdeg(cfg.RECOVERY_ARRIVAL_DEG):
                    self.controller.phase = 'held'
                elif time.monotonic() >= self.controller.recovery_deadline:
                    raise RuntimeError('机械臂当前位置保持到位超时')
            if self.ready:
                notice = '机械臂接管完成，位置保持中；可接收轨迹。'
            else:
                notice = self.controller.notice
            if notice != self.last_notice:
                self.last_notice = notice
                self.report(notice)
        except ArmRequestRejected:
            # Gamepad's take_control() only reports a notice for a refused
            # precondition. Return to monitor so the next explicit R can retry.
            self.requested = self.collecting = False
            raise
        except Exception:
            self.failed = True
            self.stop('机械臂恢复或保持失败')
            raise

    def stop(self, reason):
        self.failed = True
        if self.controller and self.controller.owns_control and self.controller.phase != 'stopped':
            self.controller.stop(reason)

    def close(self):
        try:
            if self.controller and self.controller.owns_control and not self.failed:
                self.controller.close()
        finally:
            if self.robot is not None:
                self.robot.DisconnectPort()
                self.robot = None

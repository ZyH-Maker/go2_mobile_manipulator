"""两个键盘程序的统一配置。只改本文件，保存后重启程序生效。

角度用度（deg），位置用毫米（mm），时间用秒（s）。
CAN 协议编号、六关节结构和单位换算不是调参项，保留在代码中。
"""
import math
from pathlib import Path

# ===== 常用：单关节程序 keyboard_joint.py =====
JOINT_CONTROL_SPEED_PERCENT = 10       # 底层速度，1～100
JOINT_DEFAULT_STEP_DEG = 0.5            # 启动时每字符角度
JOINT_SMALL_STEP_DEG = 0.2              # [ 键选择
JOINT_LARGE_STEP_DEG = 0.5              # ] 键选择
JOINT_KEY_INTERVAL_S = 0.2             # 两步最短间隔；0.2 = 最多 5 步/秒

# ===== 常用：末端程序 keyboard_cartesian.py =====
CONTROL_SPEED_PERCENT = 100            # 普通点动及 Home 的底层速度
SPEED_FACTOR = 1.0                     # 普通点动倍率
LINEAR_SPEED_MM_S = 50.0                # 位置目标变化率上限
JOINT_SPEED_DEG_S = 25.0                # J1/J5/J6 目标变化率上限
KEY_STEP_SECONDS = 0.02                # 字符步长 = 速度 × 倍率 × 此值，也用于限频
# 当前每字符 1 mm / 0.5°，实际长按速度受系统重复频率影响。

# ===== Home（六关节全零） =====
HOME_SPEED_DEG_S = JOINT_SPEED_DEG_S * SPEED_FACTOR  # 可直接改为独立数值
HOME_MAX_DT_S = 0.04                   # 每帧积分时间上限，不补发延迟的大步
HOME_TRACKING_TOLERANCE_DEG = 0.5       # 超过此跟踪差先等待
HOME_ARRIVAL_TOLERANCE_DEG = 0.5        # 各关节距零小于此值判定完成
HOME_PROGRESS_DEG = 0.1                # 最远关节至少接近此角度才计为有进展
HOME_STALL_TIMEOUT_S = 5.0             # 持续无进展时停止
HOME_TIMEOUT_MARGIN_S = 20.0           # 总期限 = 最大回零角度/速度 + 余量
HOME_CONTINUE_AFTER_FINISH = True       # True: 完成后直接控制；False: 暂停等 Enter

# ===== CAN 与循环 =====
CAN_INTERFACE = 'can0'                 # 可被命令行 --can 覆盖；波特率仍需系统设为 1 Mbps
STARTUP_FEEDBACK_WAIT_S = 3.0
CAN_PUMP_DURATION_S = 0.02
CAN_RECEIVE_TIMEOUT_S = 0.001
TARGET_REFRESH_INTERVAL_S = 0.02
KEY_BACKLOG_LIMIT = 256                # 一轮最多读取的字符数

# ===== 限位与运行保护：修改前核对硬件，不能靠扩大限位消除报错 =====
JOINT_LIMITS_DEG = ((-150, 150), (0, 180), (-170, 0),
                    (-100, 100), (-70, 70), (-120, 120))
FEEDBACK_TIMEOUT_S = 0.5
RANGE_TOLERANCE_DEG = 1.0              # 普通接管容许的编码器边界偏差
TRACKING_LIMIT_DEG = 2.0               # 普通目标与实际关节的最大偏差

# ===== 复位、使能及 R 越界恢复 =====
RECOVERY_SPEED_PERCENT = 5
RECOVERY_MAX_DEG = 5.0
RECOVERY_TRACKING_MARGIN_DEG = 1.0
RECOVERY_ARRIVAL_DEG = 0.5
RECOVERY_TIMEOUT_S = 10.0
MODE_EXIT_WAIT_S = 0.2
RESET_MIN_WAIT_S = 1.0
RESET_TIMEOUT_S = 5.0
RESET_STABLE_TIME_S = 0.3
RESET_STABLE_DELTA_DEG = 0.2
ENABLE_TIMEOUT_S = 4.0
ENABLE_INTERVAL_S = 0.1
ENABLE_MIN_COUNT = 2
MODE_TIMEOUT_S = 2.0

# ===== 运动学：高级参数，默认对应本机 SDK 模型 =====
DH_OFFSET = 1                         # SDK DH 偏置模型：0 或 1
IK_SUBSTEP_MM = 0.5                   # 每个字符位移拆分成的小段
IK_FRAME_TIMEOUT_S = 0.08
IK_SOLVE_TIMEOUT_S = 0.08
IK_MAX_REQUEST_MM = 2.01
IK_MAX_ITERATIONS = 12
IK_MAX_JOINT_CHANGE_DEG = 1.0
IK_ITERATION_STEP_DEG = 0.5
IK_CONVERGENCE_MM = 0.015
IK_ACCEPTANCE_MM = 0.03
IK_DIFFERENCE_STEP_DEG = 0.001         # 数值雅可比扰动量
IK_MIN_SINGULAR_VALUE = 0.0001         # 雅可比单位 mm / 0.001°
IK_DAMPING = 1e-10

# ===== 日志 =====
LOG_DIRECTORY = 'runtime/logs/arm'                # 相对项目目录，也可以填绝对路径
JOINT_LOG_FILE = 'keyboard_control.jsonl'
CARTESIAN_LOG_FILE = 'keyboard_cartesian.jsonl'
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 2
LOG_SAMPLE_INTERVAL_S = 0.25


def validate():
    """在连接 CAN 前拒绝非法配置；不静默回退到默认值。"""
    positive = [name for name in globals() if name.isupper() and name != 'JOINT_LIMITS_DEG' and
                (name.endswith(('_S', '_DEG', '_MM', '_PERCENT')) or name in (
                    'SPEED_FACTOR', 'KEY_STEP_SECONDS', 'IK_MIN_SINGULAR_VALUE',
                    'IK_DAMPING', 'KEY_BACKLOG_LIMIT', 'ENABLE_MIN_COUNT',
                    'IK_MAX_ITERATIONS', 'LOG_MAX_BYTES', 'LOG_BACKUP_COUNT'))]
    for name in positive:
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'arm_config.py: {name} 必须是有限正数')
        if name.endswith('_PERCENT') and (type(value) is not int or value > 100):
            raise ValueError(f'arm_config.py: {name} 必须是 1～100 的整数')
    for name in ('KEY_BACKLOG_LIMIT', 'ENABLE_MIN_COUNT', 'IK_MAX_ITERATIONS', 'LOG_MAX_BYTES', 'LOG_BACKUP_COUNT'):
        if type(globals()[name]) is not int:
            raise ValueError(f'arm_config.py: {name} 必须是整数')
    if type(DH_OFFSET) is not int or DH_OFFSET not in (0, 1):
        raise ValueError('arm_config.py: DH_OFFSET 必须是 0 或 1')
    if type(HOME_CONTINUE_AFTER_FINISH) is not bool:
        raise ValueError('arm_config.py: HOME_CONTINUE_AFTER_FINISH 必须为 True 或 False')
    for name in ('CAN_INTERFACE', 'LOG_DIRECTORY', 'JOINT_LOG_FILE', 'CARTESIAN_LOG_FILE'):
        if not isinstance(globals()[name], str) or not globals()[name].strip():
            raise ValueError(f'arm_config.py: {name} 必须是非空字符串')
    if not isinstance(JOINT_LIMITS_DEG, (tuple, list)) or len(JOINT_LIMITS_DEG) != 6:
        raise ValueError('arm_config.py: JOINT_LIMITS_DEG 必须包含六对限位')
    for bounds in JOINT_LIMITS_DEG:
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or any(type(x) not in (float, int) or not math.isfinite(x) for x in bounds)
                or not bounds[0] <= 0 <= bounds[1] or bounds[0] >= bounds[1]):
            raise ValueError('arm_config.py: 每对限位必须有限、下限小于上限且包含全零 Home')
    if RESET_TIMEOUT_S <= max(RESET_MIN_WAIT_S, RESET_STABLE_TIME_S):
        raise ValueError('arm_config.py: RESET_TIMEOUT_S 必须大于复位等待和稳定时间')
    if ENABLE_TIMEOUT_S <= ENABLE_INTERVAL_S * ENABLE_MIN_COUNT:
        raise ValueError('arm_config.py: ENABLE_TIMEOUT_S 太短，无法完成配置的使能次数')
    if RECOVERY_MAX_DEG < RANGE_TOLERANCE_DEG:
        raise ValueError('arm_config.py: RECOVERY_MAX_DEG 不能小于 RANGE_TOLERANCE_DEG')
    if IK_SUBSTEP_MM > IK_MAX_REQUEST_MM or IK_CONVERGENCE_MM > IK_ACCEPTANCE_MM:
        raise ValueError('arm_config.py: 逆解子步或收敛阈值超过其允许上限')


# 下方为派生值，不需要手动修改。SDK 的角度单位为 0.001°。
validate()
PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / LOG_DIRECTORY
LOCK_PATH = PROJECT_DIR / 'runtime/locks/arm.lock'  # 固定共享锁，不能按程序拆分


def mdeg(degrees):
    return round(degrees * 1000)

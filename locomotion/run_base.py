"""Go2 底盘策略入口：订阅真实机械臂状态；L1 准备，Y 启动策略。"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from base_core import validate_policy_pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True, help='Go2 DDS 网卡，例如 eth1')
    parser.add_argument('--policy-kp', type=float, default=20.0, help='策略阶段 Kp，默认 20；范围 (0, 70]')
    parser.add_argument('--policy-kd', type=float, default=1.5, help='策略阶段 Kd，默认 1.5；范围 [0, 6]')
    args = parser.parse_args()
    try:
        validate_policy_pd(args.policy_kp, args.policy_kd)
    except ValueError as exc:
        parser.error(str(exc))
    if not Path('/sys/class/net', args.interface).exists():
        parser.error('网卡不存在')
    lock = open('/tmp/go2_piper_s2r_live.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('另一个 Go2/Piper 真机程序占用控制锁')
    # Explicit interface; do not inherit an accidental loopback-only test setup.
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
    os.environ['CYCLONEDDS_URI'] = ('<CycloneDDS><Domain><General><Interfaces>'
        '<NetworkInterface name="%s"/></Interfaces></General></Domain></CycloneDDS>' % args.interface)
    muse = Path(get_package_share_directory('state_estimator'))
    out = ROOT/'runtime/logs/base'/('policy_readonly_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    commands = [
        [str(Path(get_package_prefix('go2_hal'))/'bin/managed_go2_hal')],
        ['ros2', 'run', 'state_estimator', 'state_estimator_node', '--ros-args',
         '-p', 'config_dir:='+str(muse/'config'), '-p', 'launch_dir:='+str(muse/'launch')],
    ]
    children, logs = [], []
    coordinator = None
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
        if coordinator is not None and coordinator.poll() is None:
            coordinator.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for i, command in enumerate(commands):
            log = (out/('%d.log'%i)).open('w', buffering=1)
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        command = [sys.executable, str(ROOT/'locomotion/base_controller.py'),
                   '--policy-kp', str(args.policy_kp), '--policy-kd', str(args.policy_kd),
                   '--lock-fd', str(lock.fileno()), '--output', str(out/'control.jsonl')]
        for child in children:
            command += ['--peer-pid', str(child.pid)]
        print('日志目录：', out, flush=True)
        print('准备 PD=60/5；策略 PD=%g/%g' % (args.policy_kp, args.policy_kd), flush=True)
        print('从独立机械臂执行节点读取真实状态。L1 准备 Go2；Y 启动策略。本程序不控制机械臂。', flush=True)
        coordinator = subprocess.Popen(command, pass_fds=(lock.fileno(),), start_new_session=True)
        while coordinator.poll() is None:
            if any(child.poll() is not None for child in children):
                print('必要进程已退出，正在停止协调器，请检查日志。', flush=True)
                coordinator.send_signal(signal.SIGTERM)
            time.sleep(0.1)
        if coordinator.returncode:
            raise SystemExit(coordinator.returncode)
    finally:
        # The coordinator gets its shutdown interval while the HAL remains alive.
        if coordinator is not None and coordinator.poll() is None:
            coordinator.terminate()
            try:
                coordinator.wait(timeout=4)
            except subprocess.TimeoutExpired:
                coordinator.kill()
                coordinator.wait()
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

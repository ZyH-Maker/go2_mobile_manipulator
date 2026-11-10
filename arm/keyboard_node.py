"""Shared ROS backend for the joint and Cartesian keyboard target nodes."""
import argparse
import json
import math
import os
import select
import signal
import sys
import termios
import time
import tty
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from arm_common import ARM_NAMES, named_arm
from keyboard_targets import KeyboardTargets, JointKeyboardTargets, MOTION_KEYS
from arm_trajectory import MAX_SPEED_RAD_S
import arm_config as cfg
from keyboard_display import screen_lines, draw_screen

PREFIX='/follower/arm_controller'


def joint_message(target, seconds):
    msg=JointTrajectory();msg.joint_names=ARM_NAMES
    point=JointTrajectoryPoint();point.positions=np.asarray(target,dtype=float).tolist()
    ns=max(1,math.ceil(seconds*1e9))
    point.time_from_start.sec=ns//1000000000;point.time_from_start.nanosec=ns%1000000000
    msg.points=[point]
    return msg


class ArmKeyboard(Node):
    def __init__(self, mode='cartesian'):
        if mode not in ('joint','cartesian'):raise ValueError('未知键盘模式')
        super().__init__('piper_arm_keyboard')
        self.mode=mode
        self.pub=self.create_publisher(JointTrajectory,PREFIX+'/joint_trajectory',1)
        self.create_subscription(JointState,'/follower/feedback/joint_states',self.on_feedback,1)
        self.create_subscription(String,PREFIX+'/status',self.on_status,1)
        self.service_clients={name:self.create_client(Trigger,PREFIX+'/'+name) for name in ('enable','recover','hold','stop')}
        self.actual=None;self.feedback_at=-math.inf;self.status_at=-math.inf
        self.status={};self.generator=JointKeyboardTargets() if mode=='joint' else KeyboardTargets();self.pending=None
        self.paused=True;self.homing=False;self.inflight=None
        self.command_at=0.;self.command_duration=.02;self.start_q=None
        self.last_phase=None;self.last_error=None;self.service_generation=0
        self.notice='等待执行节点和实际角度反馈；Enter 接管，R 越界恢复'

    def notify(self, message):
        self.notice=str(message)

    def on_feedback(self,msg):
        try:
            q,_=named_arm(msg.name,msg.position,msg.velocity)
            self.actual=q;self.feedback_at=time.monotonic()
        except (ValueError,IndexError):self.feedback_at=-math.inf

    def on_status(self,msg):
        try:
            status=json.loads(msg.data)
            if type(status.get('ready')) is not bool or not isinstance(status.get('phase'),str):return
            self.status=status;self.status_at=time.monotonic()
            phase=status['phase']
            self.last_phase=phase
            if status.get('error') and status.get('error')!=self.last_error:
                self.pending=None;self.homing=False
                self.notify(status['error'])
            self.last_error=status.get('error')
            if not status['ready']:
                self.pending=None;self.homing=False
            if self.homing and phase=='held' and self.actual is not None and np.max(np.abs(self.actual)) <= math.radians(.5):
                self.homing=False;self.pending=None
                self.notify('Home 完成，可直接继续运动')
        except (ValueError,KeyError,TypeError):return

    def ready(self):
        now=time.monotonic()
        return self.actual is not None and now-self.feedback_at < .1 and now-self.status_at < .2 and self.status.get('ready',False)

    def service(self,name):
        if name not in ('hold','stop') and self.inflight is not None and not self.inflight.done():
            self.notify('上一服务请求尚未返回');return
        client=self.service_clients[name]
        if not client.service_is_ready():
            self.notify('执行节点服务不可用：'+name);return
        self.service_generation+=1
        generation=self.service_generation
        self.inflight=client.call_async(Trigger.Request())
        def done(future):
            try:
                if generation!=self.service_generation:return
                result=future.result()
                self.notify(result.message)
                if result.success and name in ('enable','recover'):
                    self.paused=False;self.pending=None
            except Exception as exc:self.notify(str(exc))
        self.inflight.add_done_callback(done)

    def cancel(self, emergency=False):
        self.paused=True;self.homing=False;self.pending=None
        self.service('stop' if emergency else 'hold')

    def key(self,key):
        if key==' ':self.cancel(True);return
        if key=='p':self.cancel();return
        if self.mode=='joint':
            if key=='e':key='\n'
            elif key in '123456' and len(key)==1:
                self.generator.selected=int(key)-1
                self.report_joint_selection();return
            elif key in ('[',']'):
                self.generator.step_rad=math.radians(cfg.JOINT_SMALL_STEP_DEG if key=='[' else cfg.JOINT_LARGE_STEP_DEG)
                self.report_joint_selection();return
        if key in ('\n','\r'):
            if self.ready():
                self.paused=False;self.pending=None
                self.notify('已继续，可用运动键控制机械臂')
            else:self.service('enable')
            return
        if key=='r':self.service('recover');return
        if self.paused or not self.ready():return
        if key=='b' and self.mode=='cartesian':
            if self.homing:return
            start=self.actual.copy()
            # Include the last commanded endpoint as well as measured pose.
            # The executor may still be interpolating when Home arrives.
            extent=float(np.max(np.abs(start)))
            if self.pending is not None:extent=max(extent,float(np.max(np.abs(self.pending))))
            duration=max(.04,extent/MAX_SPEED_RAD_S+.04)
            self.pub.publish(joint_message(np.zeros(6),duration))
            self.homing=True;self.pending=None;self.generator.last_character=time.monotonic()
            self.notify('正在回 Home 零位；P 取消并保持，空格急停')
            return
        motion_keys=('j','l') if self.mode=='joint' else MOTION_KEYS
        if key not in motion_keys or self.homing:return
        now=time.monotonic()
        # Start a replacement from the interpolated command, not a jumped
        # previous endpoint. Do not accumulate a backlog of key repeats.
        if self.pending is None:start=self.actual.copy()
        else:
            u=np.clip((now-self.command_at)/self.command_duration,0.,1.)
            start=self.start_q*(1-u)+self.pending*u
        result=self.generator.step(key,start,self.actual,now)
        if result is None:return
        # IK can take time: check actual feedback freshness again before send.
        if not self.ready():return
        q,duration=result
        # Zero-stamped trajectories begin upon receipt. Duration includes a
        # small transport margin so adjacent replacements stay within 25°/s.
        duration=max(duration,.04)
        self.pub.publish(joint_message(q,duration))
        self.pending=q;self.start_q=start;self.command_at=time.monotonic();self.command_duration=duration

    def report_joint_selection(self):
        i=self.generator.selected
        measured='；等待实际角度反馈' if self.actual is None else '；实际角度 %.3f°'%math.degrees(self.actual[i])
        self.notify('选中 J%d；步长 %.1f°%s'%(i+1,math.degrees(self.generator.step_rad),measured))


def handle_input(node, data):
    """Drain key repeats without queuing motion; stop has highest priority."""
    data=data.lower()
    if ' ' in data:node.cancel(True)
    elif any(key in data for key in ('x','\x03','\x04')):return False
    elif 'p' in data:node.cancel()
    elif data:node.key(data[-1])
    return True


def main(argv=None, mode='cartesian'):
    description='单关节键盘目标节点；需先启动 run_executor.py' if mode=='joint' else '末端键盘目标节点；需先启动 run_executor.py'
    parser=argparse.ArgumentParser(description=description);parser.parse_args(argv)
    if not sys.stdin.isatty():parser.error('需要交互终端')
    rclpy.init();node=ArmKeyboard(mode);exiting=[False]
    def stop(signum,frame):exiting[0]=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    terminal=termios.tcgetattr(sys.stdin)
    if mode=='joint':node.report_joint_selection()
    try:
        # Alternate screen restores the user's terminal on normal exit.
        sys.stdout.write('\x1b[?1049h\x1b[?25l');sys.stdout.flush()
        tty.setcbreak(sys.stdin.fileno())
        last_draw=-math.inf
        while rclpy.ok() and not exiting[0]:
            rclpy.spin_once(node,timeout_sec=.01)
            if select.select([sys.stdin],[],[],0)[0]:
                data=os.read(sys.stdin.fileno(),256).decode('ascii',errors='ignore').lower()
                if not handle_input(node,data):break
            now=time.monotonic()
            if now-last_draw>=.1:
                draw_screen(screen_lines(node,now));last_draw=now
    finally:
        termios.tcsetattr(sys.stdin,termios.TCSADRAIN,terminal)
        sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()
        node.cancel()
        until=time.monotonic()+.3
        while rclpy.ok() and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
        print('键盘已退出；'+node.notice,flush=True)
        node.destroy_node();rclpy.shutdown()

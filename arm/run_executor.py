"""Standalone PiPER position executor with the author's JointTrajectory topic.

Startup receives actual CAN state only. enable/recover services explicitly take
control. No Go2 command is ever sent by this process.
"""
import argparse
import fcntl
import json
import math
import signal
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
from arm_backend import ArmBackend, ArmRequestRejected
from arm_trajectory import JointTrajectoryPlan
from arm_common import ARM_NAMES
import arm_config as cfg

PREFIX = '/follower/arm_controller'
FEEDBACK_TOPIC = '/follower/feedback/joint_states'
TRAJECTORY_TOPIC = PREFIX+'/joint_trajectory'


class ArmExecutor(Node):
    def __init__(self, channel='can0', backend=None, output=None):
        super().__init__('piper_arm_executor')
        self.backend = backend or ArmBackend(channel, self.report)
        self.plan = None
        self.hal_active = False
        self.last_error = ''
        self.last_q = self.last_feedback_at = None
        self.output = output.open('a', buffering=1) if output else None
        self.last_status_at = 0.
        self.feedback_pub = self.create_publisher(JointState, FEEDBACK_TOPIC, 1)
        self.status_pub = self.create_publisher(String, PREFIX+'/status', 1)
        self.create_subscription(Bool,'/locopilot/hal_active',self.on_hal,1)
        self.create_subscription(JointTrajectory,TRAJECTORY_TOPIC,self.on_trajectory,1)
        self.create_service(Trigger,PREFIX+'/recover',lambda req,res:self.take_control(res,True))
        self.create_service(Trigger,PREFIX+'/enable',lambda req,res:self.take_control(res,False))
        self.create_service(Trigger,PREFIX+'/hold',self.hold)
        self.create_service(Trigger,PREFIX+'/stop',self.stop)
        self.backend.connect()
        self.timer = self.create_timer(.005,self.tick)

    def report(self, message):
        self.get_logger().info(message)
        self.record('event',message=message)

    def record(self, kind, **fields):
        if self.output:
            self.output.write(json.dumps(dict(kind=kind,at=time.monotonic(),**fields),ensure_ascii=False)+'\n')

    def on_hal(self, msg):
        self.hal_active = msg.data

    def take_control(self, response, recover):
        if self.hal_active:
            response.success=False;response.message='底盘已接管，不允许此时复位/重新使能机械臂'
            return response
        if self.backend.ready:
            response.success=True;response.message='机械臂已接管，位置保持/轨迹执行中'
            return response
        try:
            self.backend.request(recover=recover)
            self.last_error=''
            response.success=True;response.message='已接受恢复请求；等待 status.ready=true 后再发送轨迹' if recover else '已接受当前位置接管请求；等待完成'
        except Exception as exc:
            response.success=False;response.message=str(exc)
        return response

    def hold(self, request, response):
        self.plan=None
        try:
            self.backend.hold()
            response.success=not self.backend.failed
            if self.backend.failed:
                self.last_error=self.backend.controller.notice if self.backend.controller else '停止已锁定'
                response.message=self.last_error
            else:
                self.last_error=''
                response.message='已取消轨迹/接管请求；已接管时保持当前位置'
        except Exception as exc:
            self.fail(str(exc));response.success=False;response.message=str(exc)
        return response

    def stop(self, request, response):
        self.plan=None
        try:
            self.backend.stop('操作者请求机械臂快速停止')
            self.last_error='操作者请求机械臂快速停止'
            response.success=True;response.message='机械臂停止已锁定；需要重启执行节点'
        except Exception as exc:
            response.success=False;response.message=str(exc)
        return response

    def on_trajectory(self, message):
        try:
            # Process current feedback before accepting/replacing a command.
            self.backend.advance()
            if not self.backend.ready:
                raise ValueError('机械臂未接管到位，请先 enable 或 recover')
            if self.count_publishers(TRAJECTORY_TOPIC) != 1:
                raise ValueError('轨迹来源冲突：一次只运行一个键盘或策略发布者')
            now=time.monotonic()
            stamp=message.header.stamp.sec+message.header.stamp.nanosec/1e9
            offset=stamp-self.get_clock().now().nanoseconds/1e9 if stamp else 0.
            if not math.isfinite(offset) or offset < -.1:
                raise ValueError('轨迹起始时间已过期；立即执行可使用零时间戳')
            current=(self.plan.sample(now) if self.plan else np.deg2rad(np.asarray(self.backend.controller.target)/1000.))
            replacement=JointTrajectoryPlan(message,current,now+max(offset,0.))
            self.plan=replacement;self.last_error=''
            self.record('trajectory',names=list(message.joint_names),times=replacement.times,
                        positions=[q.tolist() for q in replacement.positions])
        except Exception as exc:
            self.last_error=str(exc)
            self.get_logger().warning('轨迹未接受：'+str(exc))
            self.record('rejected',reason=str(exc))
            if self.backend.failed:self.plan=None

    def fail(self, reason):
        self.plan=None;self.last_error=reason
        try:self.backend.stop(reason)
        except Exception as exc:self.get_logger().error('机械臂停止发送失败：'+str(exc))
        self.get_logger().error(reason)
        self.record('fault',reason=reason)

    def publish_actual(self, now):
        f=self.backend.feedback
        if f is None or any(now-f.last_seen.get(i,-float('inf')) >= .08 for i in (0x2a5,0x2a6,0x2a7)):
            return
        q=np.deg2rad(np.asarray(f.joints(),dtype=float)/1000.)
        dq=np.zeros(6) if self.last_q is None else (q-self.last_q)/max(now-self.last_feedback_at,1e-6)
        self.last_q=q;self.last_feedback_at=now
        msg=JointState();msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=ARM_NAMES;msg.position=q.tolist();msg.velocity=dq.tolist()
        self.feedback_pub.publish(msg)

    def tick(self):
        now=time.monotonic()
        try:
            if self.backend.requested and not self.backend.failed:self.backend.advance()
            else:self.backend.poll()
            if self.plan:
                if self.count_publishers(TRAJECTORY_TOPIC) != 1:
                    self.plan=None;self.backend.hold()
                    self.last_error='轨迹发布者退出或冲突，已取消轨迹并保持当前位置'
                elif self.backend.ready:
                    self.backend.command(self.plan.sample(now))
                    if now >= self.plan.end_time:self.plan=None
            self.publish_actual(now)
        except ArmRequestRejected as exc:
            self.last_error=str(exc)
            self.get_logger().warning('接管请求未接受，保持只读：'+str(exc))
            self.record('rejected',operation='take_control',reason=str(exc))
        except Exception as exc:self.fail(str(exc))
        if now-self.last_status_at >= .05:
            self.last_status_at=now
            c=self.backend.controller
            msg=String();msg.data=json.dumps({'ready':self.backend.ready,
                'phase':'fault' if self.backend.failed else ('trajectory' if self.plan else (c.phase if c else 'monitor')),
                'error':self.last_error,'actual_position':None if self.last_q is None else self.last_q.tolist(),
                'target_position':None if c is None or c.target is None else np.deg2rad(np.asarray(c.target)/1000.).tolist()},ensure_ascii=False)
            self.status_pub.publish(msg)

    def close(self):
        try:
            self.plan=None
            # Normal exit holds measured position, as in Gamepad. No disable.
            self.backend.close()
        finally:
            if self.output:self.output.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can',default='can0')
    args=parser.parse_args()
    # Same file as the original direct-CAN Gamepad programs, even across the
    # host/container bind mount. They cannot own the same arm concurrently.
    cfg.LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
    lock=open(cfg.LOCK_PATH,'a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init();exiting=[False]
    def stop(signum,frame):exiting[0]=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    out=Path(__file__).resolve().parents[1]/'runtime/logs/arm'/('arm_executor_'+time.strftime('%Y%m%d_%H%M%S')+'.jsonl')
    out.parent.mkdir(parents=True,exist_ok=True)
    node=ArmExecutor(args.can,output=out)
    print('机械臂执行节点：启动只读；键盘 Enter 接管、R 恢复；轨迹接口 '+TRAJECTORY_TOPIC,flush=True)
    try:
        while rclpy.ok() and not exiting[0]:rclpy.spin_once(node,timeout_sec=.005)
    finally:
        node.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()

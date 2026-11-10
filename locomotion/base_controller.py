"""Go2 policy coordinator; the arm is observed only and never commanded."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import sys
import time

# Load Torch/OpenMP before MuJoCo and ROS consume aarch64's static TLS slots.
import torch
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from unitree_go.msg import LowState
from unitree_api.msg import Request, Response
from dls2_interface.msg import BaseState, BlindState, TrajectoryGenerator


from base_core import (ARM_NAMES, LEG_NAMES, L1, Y, B, L2, R2, Lifecycle,
                       vector, named_arm, remote, neutral,
                       check_feedback, clamp_leg_target, arm_limits_from_urdf, validate_policy_pd)

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT/'third_party/locomotion_policy/deploy'
sys.path[:0] = [str(DEPLOY), str(DEPLOY/'mujoco')]
from locomotion_policy_wrapper import LocomotionPolicyWrapper
import mujoco_utils


class PolicyController(Node):
    def __init__(self, output, peer_pids=(), policy_kp=20.0, policy_kd=1.5):
        self.policy_kp, self.policy_kd = validate_policy_pd(policy_kp, policy_kd)
        super().__init__('locopilot_policy')
        self.life = Lifecycle()
        self.peer_pids = peer_pids
        self.hal_ack = False
        self.release_confirmed = False
        self.release_confirmed_at = None
        self.hal_enable_at = None
        self.output = output
        self.events = queue.Queue()
        self.receipts = {}
        self.raw = self.base = self.blind = self.arm = None
        self.old_keys = None
        self.axes = np.zeros(3)
        self.keys = 0
        self.command = np.zeros(3)
        self.hal_active = False
        self.rpc = self.service_future = None
        self.started = self.dwell = None
        self.leg_start = None
        self.fault_stop_sent = False
        self.model = mujoco.MjModel.from_xml_path(str(DEPLOY/'mujoco/models/scene_flat.xml'))
        self.data = mujoco.MjData(self.model)
        self.policy = LocomotionPolicyWrapper(self.model)
        # Prewarm ONNX before creating any control publishers or timers.
        self.policy.policy.run(None, {self.policy.policy.get_inputs()[0].name:
                                     np.zeros((1, 300), dtype=np.float32)})
        self.joint_limits = np.array([self.model.jnt_range[self.model.joint(n).id] for n in LEG_NAMES])
        # Match the author's deployment: virtual PD targets are limited by
        # torque, not clipped to the model's physical joint-position range.
        self.leg_target_limits = None
        standard_urdf = ROOT/'models/piper_limits.urdf'
        self.arm_limits = arm_limits_from_urdf(standard_urdf)
        self.arm_limit_source = str(standard_urdf.relative_to(ROOT))
        self.get_logger().info('机械臂反馈限位使用标准版 Piper URDF；底盘策略仍为原 Piper L 训练模型')
        self.torque = self.model.actuator_ctrlrange[:12, 1]*0.95
        self.stream = output.open('w', buffering=1)
        self.stream.write(json.dumps({'kind':'start','arm_target_rad':None,
            'policy':'arm_moving_clock','stand_PD':[60,5],'policy_PD':[self.policy_kp,self.policy_kd],
            'leg_target_limit':'original_pd_torque_only',
            'arm_limit_source':self.arm_limit_source,'arm_limits_rad':self.arm_limits.tolist(),
            'arm_observation_source':'can_joint_feedback','hardware_validated':False})+'\n')
        self.create_subscription(LowState,'/lowstate',self.on_low,qos_profile_sensor_data)
        self.create_subscription(BaseState,'/base_state',lambda m:self.receive('base',m),1)
        self.create_subscription(BlindState,'/blind_state',lambda m:self.receive('blind',m),1)
        self.create_subscription(JointState,'/follower/feedback/joint_states',self.on_arm,1)
        self.create_subscription(Bool,'/locopilot/hal_active',self.on_hal,1)
        self.api_pub = self.create_publisher(Request,'/api/motion_switcher/request',1)
        self.create_subscription(Response,'/api/motion_switcher/response',self.on_response,1)
        self.leg_pub = self.create_publisher(TrajectoryGenerator,'/trajectory_generator',1)
        self.enable_hal = self.create_client(SetBool,'/locopilot/arm_lowlevel')
        self.stop_hal = self.create_client(SetBool,'/locopilot/arm_lowlevel')
        self.timer = self.create_timer(0.02,self.tick)
        self.last_phase = None
        self.wait_report_at = 0.0

    def receive(self, name, msg):
        setattr(self,name,msg)
        self.receipts[name] = time.monotonic()

    def on_arm(self,msg):
        try:
            named_arm(msg.name,msg.position,msg.velocity)
        except (ValueError,IndexError):
            self.life.fault('机械臂反馈格式或关节名称无效')
            return
        self.receive('arm',msg)

    def on_low(self,msg):
        self.receive('raw',msg)
        try:
            keys, axes = remote(msg.wireless_remote)
        except ValueError as exc:
            self.receipts.pop('remote',None)
            self.axes=np.zeros(3);self.keys=0;self.old_keys=None
            if self.life.phase != 'WAIT':
                self.life.fault('遥控器反馈无效或不可用：'+str(exc))
            return
        self.receipts['remote']=time.monotonic()
        self.keys,self.axes = keys,axes
        if keys & (L2|R2):
            self.events.put('stop')
        if self.old_keys is not None:
            edges = keys & ~self.old_keys
            for mask,event in [(L1,'prepare'),(Y,'rl'),(B,'pause')]:
                if edges & mask:
                    self.events.put(event)
        self.old_keys = keys

    def on_hal(self,msg):
        self.hal_active = msg.data
        self.receipts['hal'] = time.monotonic()

    def call_service(self,client,value,label):
        if not client.service_is_ready():
            raise ValueError('服务不可用：'+label)
        if label=='hal_enable':self.hal_enable_at=time.monotonic()
        req = SetBool.Request(); req.data = value
        self.service_future = (client.call_async(req),time.monotonic(),label)

    def service_done(self):
        future,at,label = self.service_future
        if future.done():
            response = future.result()
            if response is None or not response.success:
                raise ValueError('服务拒绝：'+label+'；'+(response.message if response is not None else '未收到响应'))
            self.service_future = None
            return label
        if time.monotonic()-at > 6:
            raise ValueError('服务响应超时：'+label)
        return None

    def api(self,api_id):
        req = Request()
        req.header.identity.id = time.time_ns()
        req.header.identity.api_id = api_id
        self.rpc = [req.header.identity.id, api_id, time.monotonic(), None]
        self.api_pub.publish(req)

    def on_response(self,msg):
        if self.rpc and msg.header.identity.id == self.rpc[0]:
            self.rpc[3] = msg

    def validate(self):
        required=['raw','remote','base','blind','hal']
        required += ['arm']
        for name in required:
            try:
                check_feedback(time.monotonic(),[self.receipts.get(name,-float('inf'))])
            except ValueError:
                raise ValueError('遥控器不可用：请开启并配对遥控器，将摇杆回中' if name=='remote' else '反馈超时：'+name)
        q,dq = named_arm(self.arm.name,self.arm.position,self.arm.velocity)
        legs = vector(self.blind.joints_position,12)
        speeds = vector(self.blind.joints_velocity,12)
        quat = vector(self.base.pose.orientation,4)
        quat_norm=float(np.linalg.norm(quat))
        if quat_norm < 1e-6:
            raise ValueError('姿态四元数为零，无效')
        quat=quat/quat_norm
        vector(self.base.pose.position,3)
        vector(self.base.velocity.linear,3); vector(self.base.velocity.angular,3)
        self.data.qpos[:3] = self.base.pose.position
        self.data.qpos[3:7] = np.roll(quat,1)
        self.data.qpos[7:19] = legs; self.data.qpos[19:25] = q
        self.data.qvel[:3] = self.base.velocity.linear
        self.data.qvel[3:6] = self.base.velocity.angular
        self.data.qvel[6:18] = speeds
        self.data.qvel[18:24] = dq
        mujoco.mj_forward(self.model,self.data)
        if ((q < self.arm_limits[:,0]-0.02).any() or (q > self.arm_limits[:,1]+0.02).any()):
            raise ValueError('机械臂位置超出标准版 Piper 范围')
        if ((legs < self.joint_limits[:,0]-0.02).any() or (legs > self.joint_limits[:,1]+0.02).any()):
            raise ValueError('腿部位置超出模型范围')
        if self.life.phase in ['STAND','READY','POLICY'] and not self.hal_active:
            raise ValueError('HAL 已停止或看门狗故障已锁定')
        return q,dq,legs,speeds

    def publish_legs(self,q,kp,kd):
        msg = TrajectoryGenerator(); msg.timestamp = float(self.get_clock().now().nanoseconds)
        msg.joints_position = vector(q,12).tolist(); msg.joints_velocity = [0.0]*12
        msg.kp = [float(kp)]*12; msg.kd = [float(kd)]*12
        self.leg_pub.publish(msg)

    def infer(self):
        ref_lin,ref_ang = mujoco_utils.target_base_vel(self.data,np.r_[self.command[:2],0.0],self.command[2],frame='world')
        return self.policy.compute_control(
            base_pos=self.data.qpos[:3].copy(),base_ori_euler_xyz=mujoco_utils.base_ori_euler_xyz(self.data),
            base_quat_wxyz=self.data.qpos[3:7].copy(),base_lin_vel=mujoco_utils.base_lin_vel(self.data,'base'),
            base_ang_vel=mujoco_utils.base_ang_vel(self.data,'base'),
            heading_orientation_SO3=mujoco_utils.heading_orientation_SO3(self.data),
            joints_pos_leg=self.data.qpos[7:19].copy(),joints_vel_leg=self.data.qvel[6:18].copy(),
            ref_base_lin_vel=ref_lin,ref_base_ang_vel=ref_ang,ref_pose_command=np.zeros(2),
            joints_pos_arm=self.data.qpos[19:25].copy())

    def tick(self):
        now=time.monotonic()
        try:
            # This process never controls the arm, including on shutdown.
            while not self.events.empty():
                event=self.events.get_nowait()
                if event=='prepare' and self.life.phase=='WAIT':
                    try:
                        q,dq,legs,speeds=self.validate()
                        if not neutral(self.axes):
                            raise ValueError('准备前请将摇杆回中')
                    except ValueError as exc:
                        self.get_logger().warning('准备条件未满足，继续等待：'+str(exc))
                        continue
                    if not self.enable_hal.service_is_ready():
                        self.get_logger().warning('准备条件未满足，继续等待：HAL 接管服务不可用')
                        continue
                    if self.count_publishers('/trajectory_generator') != 1:
                        self.get_logger().warning('准备条件未满足，继续等待：腿部轨迹发布者缺失或冲突')
                        continue
                    self.leg_start=legs.copy()
                    self.life.phase='RELEASE';self.started=now
                    self.publish_legs(self.leg_start,60,5)
                    self.stream.write(json.dumps({'kind':'handover_start','at':now,'leg_start':self.leg_start.tolist()})+'\n')
                    self.api(1003)
                    self.get_logger().info('正在释放 Go2 原控制器，接管后站立，再等待人工启动策略')
                    continue
                elif event=='rl' and self.life.phase=='READY' and neutral(self.axes):
                    self.policy._observation_history.fill(0); self.policy.past_rl_actions.fill(0)
                    self.policy.phase_signal=np.array([0.,0.5,0.5,0.])
                    self.life.action(event,True)
                else:
                    self.life.action(event,neutral(self.axes))
            if self.life.phase=='WAIT':
                if now-self.wait_report_at>2:
                    self.wait_report_at=now
                    try:
                        q,_,_,_=self.validate()
                        self.get_logger().info('等待中，反馈正常；'+('机械臂角度（弧度）='+str(q.tolist()))+
                                               '；按 L1 开始 Go2 准备')
                    except ValueError as exc:
                        self.get_logger().info('等待中，条件未满足：'+str(exc))
                return
            if self.life.phase=='FAULT':
                self.stop_outputs()
                return
            for pid in self.peer_pids:
                try:
                    os.kill(pid, 0)
                    if Path('/proc/%d/stat'%pid).read_text().split()[2] == 'Z':
                        raise ProcessLookupError()
                except (ProcessLookupError, FileNotFoundError):
                    raise ValueError('必要控制进程已退出：%d'%pid)
            q,dq,legs,speeds=self.validate()
            phase=self.life.phase
            if phase=='RELEASE':
                # Hold the pre-release snapshot; do not chase a sagging pose.
                self.publish_legs(self.leg_start,60,5)
                if self.rpc:
                    if now-self.rpc[2]>3:
                        raise ValueError('原控制器切换请求超时')
                    response=self.rpc[3]
                    if response is not None:
                        api_id=self.rpc[1];self.rpc=None
                        if response.header.status.code!=0:
                            raise ValueError('原控制器拒绝切换请求')
                        if api_id==1003:
                            self.api(1001)
                        else:
                            if json.loads(response.data).get('name')!='':
                                raise ValueError('原控制器仍处于活动状态')
                            self.release_confirmed=True
                            self.release_confirmed_at=now
                elif self.service_future:
                    self.hal_ack = self.service_done()=='hal_enable'
                # DDS endpoint existence is not evidence of command traffic.
                # Pre-release Sport packets must not delay activation.
                # HAL monitors conflicting LowCmd traffic once active.
                if self.release_confirmed and not self.hal_ack and self.service_future is None:
                    self.call_service(self.enable_hal,True,'hal_enable')
                if self.hal_ack and self.hal_active:
                    self.stream.write(json.dumps({'kind':'handover_active','at':now,
                        'release_confirmed_at':self.release_confirmed_at,'hal_enable_at':self.hal_enable_at,
                        'leg_start':self.leg_start.tolist(),'leg_q':legs.tolist()})+'\n')
                    self.started=now; self.dwell=None
                    self.life.action('lowlevel_confirmed')
                elif self.hal_ack and now-self.hal_enable_at>8:
                    raise ValueError('等待 HAL 接管状态超时')
            elif phase=='STAND':
                alpha=np.clip((now-self.started)/5.,0,1)
                target=(1-alpha)*self.leg_start+alpha*self.policy.default_joint_pos_leg
                sent_target=clamp_leg_target(target,legs,speeds,60,5,self.torque,self.leg_target_limits)
                self.publish_legs(sent_target,60,5)
                error=float(np.max(np.abs(legs-target)))
                speed=float(np.max(np.abs(speeds)))
                self.stream.write(json.dumps({'kind':'stand','at':now,'alpha':float(alpha),
                    'leg_q':legs.tolist(),'leg_dq':speeds.tolist(),'stand_target':target.tolist(),
                    'sent_target':sent_target.tolist(),'max_error_rad':error,'max_speed_rad_s':speed})+'\n')
                if now-getattr(self,'stand_report_at',0)>1:
                    self.stand_report_at=now
                    index=int(np.argmax(np.abs(legs-target)))
                    self.get_logger().info('站立进度 %.0f%%；最大误差 %.3f rad，关节 %s；最大速度 %.3f rad/s' % (100*alpha,error,LEG_NAMES[index],speed))
                ready=alpha==1 and error<0.30 and speed<0.2
                self.dwell=now if ready and self.dwell is None else (self.dwell if ready else None)
                if self.dwell is not None and now-self.dwell>=0.5:
                    self.life.action('stand_ready')
                elif now-self.started>13:
                    raise ValueError('站立到位超时：最大误差 %.3f rad（要求 <0.30），最大速度 %.3f rad/s（要求 <0.2）' % (error,speed))
            elif phase=='READY':
                self.publish_legs(clamp_leg_target(self.policy.default_joint_pos_leg,legs,speeds,60,5,self.torque,self.leg_target_limits),60,5)
            elif phase=='POLICY':
                desired=np.array([self.axes[1]*0.2,self.axes[0]*0.15,self.axes[2]*0.3])
                self.command=np.zeros(3) if self.life.paused or self.keys&B else 0.7*self.command+0.3*desired
                target=clamp_leg_target(self.infer(),legs,speeds,self.policy_kp,self.policy_kd,self.torque,self.leg_target_limits)
                self.publish_legs(target,self.policy_kp,self.policy_kd)
                self.stream.write(json.dumps({'kind':'policy','at':now,'command':self.command.tolist(),
                    'arm_q':q.tolist(),'leg_target':target.tolist(),
                    'observation':self.policy._observation_history.flatten().tolist()})+'\n')
            if self.last_phase!=self.life.phase:
                self.get_logger().info('阶段：'+{'WAIT':'等待','RELEASE':'释放原控制器','STAND':'插值站立','READY':'站立待命，按 Y 启动策略','POLICY':'强化学习策略运行','FAULT':'故障'}.get(self.life.phase,self.life.phase))
                self.stream.write(json.dumps({'kind':'phase','at':now,'phase':self.life.phase})+'\n')
                self.last_phase=self.life.phase
        except Exception as exc:
            self.life.fault(exc);self.get_logger().error(str(exc));self.stop_outputs()

    def stop_outputs(self):
        # No low-level publication ever occurred if the bridge was never armed.
        if self.last_phase != 'FAULT':
            self.get_logger().error('故障已锁定：'+self.life.reason+'；需要退出并重新启动')
            self.last_phase = 'FAULT'
        if not self.fault_stop_sent and self.stop_hal.service_is_ready():
            req=SetBool.Request();req.data=False;self.stop_hal.call_async(req)
            self.fault_stop_sent=True
            self.stream.write(json.dumps({'kind':'stop','reason':self.life.reason})+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/logs/base'/('policy_%s.jsonl'%time.strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--policy-kp', type=float, default=20.0)
    parser.add_argument('--policy-kd', type=float, default=1.5)
    parser.add_argument('--lock-fd',type=int)
    parser.add_argument('--peer-pid',type=int,action='append',default=[])
    args=parser.parse_args()
    try:
        validate_policy_pd(args.policy_kp, args.policy_kd)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.lock_fd is None:
        lock=open('/tmp/go2_piper_s2r_live.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    else:
        lock=os.fdopen(args.lock_fd,'a')
    rclpy.init()
    exiting=[False]
    def stop(signum,frame):exiting[0]=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    node=PolicyController(args.output,args.peer_pid,args.policy_kp,args.policy_kd)
    print('遥控操作：L1 准备；Y 启动/恢复策略；B 速度归零；L2 或 R2 停止。Ctrl+C 退出程序。', flush=True)
    try:
        while rclpy.ok() and not exiting[0]:
            rclpy.spin_once(node,timeout_sec=0.02)
    finally:
        node.life.fault('操作者退出')
        until=time.monotonic()+1.0
        while rclpy.ok() and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=0.02)
        node.stream.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()

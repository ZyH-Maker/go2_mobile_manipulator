"""Loopback-only synthetic integration; never creates a CAN driver or a real RPC."""
import os
# Set before importing/initializing any ROS component. No network interfaces other than lo.
os.environ['ROS_DOMAIN_ID']='231'
os.environ['ROS_LOCALHOST_ONLY']='1'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
import copy
from collections import deque
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import signal
import struct
import subprocess
import tempfile
import time
from unittest.mock import Mock

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_prefix
from sensor_msgs.msg import JointState
from unitree_go.msg import LowCmd,LowState
from unitree_api.msg import Request,Response
from dls2_interface.msg import BaseState
from base_core import ARM_NAMES, L2
from base_controller import PolicyController,ROOT


def main():
    recent_lowcmd='--recent-lowcmd' in sys.argv
    stand_offset='--stand-offset' in sys.argv
    use_arm_executor='--arm-executor' in sys.argv
    rclpy.init()
    temp=Path(tempfile.mkdtemp(prefix='locopilot_test_'))
    log=(temp/'hal.log').open('w')
    executor=SingleThreadedExecutor()
    fake=Node('synthetic_hardware');node=PolicyController(temp/'control.jsonl',policy_kp=30.,policy_kd=2.)
    hal_command=[str(Path(get_package_prefix('go2_hal'))/'bin/managed_go2_hal')]
    hal=subprocess.Popen(hal_command,stdout=log,stderr=subprocess.STDOUT)
    for n in [fake,node]:
        if n is not None:executor.add_node(n)
    low_pub=fake.create_publisher(LowState,'/lowstate',qos_profile_sensor_data)
    base_pub=fake.create_publisher(BaseState,'/base_state',1)
    arm_pub=fake.create_publisher(JointState,'/follower/feedback/joint_states',1)
    low_commands=deque(maxlen=4);requests=[]
    # Synthetic Sport packets use the default header; only retain HAL output.
    fake.create_subscription(LowCmd,'/lowcmd',lambda m:low_commands.append(m) if list(m.head)==[0xFE,0xEF] else None,1)
    sport_pub=fake.create_publisher(LowCmd,'/lowcmd',1)
    sport_remove_at=[None]
    api_pub=fake.create_publisher(Response,'/api/motion_switcher/response',1)
    def respond(req):
        if req.header.identity.api_id==1001:
            assert not low_commands, 'HAL output before motion release confirmation'
        requests.append(req.header.identity.api_id)
        if recent_lowcmd and req.header.identity.api_id==1003:
            # Sport's final command just before it acknowledges release must
            # not require a 0.5 s unsupported interval before HAL can take over.
            sport_pub.publish(LowCmd())
        if req.header.identity.api_id==1001:sport_remove_at[0]=time.monotonic()+.5
        out=Response();out.header.identity=req.header.identity
        out.header.status.code=0;out.data=json.dumps({'name':''});api_pub.publish(out)
    fake.create_subscription(Request,'/api/motion_switcher/request',respond,1)
    raw=LowState();raw.bms_state.soc=80
    raw.wireless_remote=[0]*40;raw.wireless_remote[0]=0x55;raw.wireless_remote[1]=0x51
    raw.imu_state.quaternion=[1.,0.,0.,0.]
    mapping=[3,4,5,0,1,2,9,10,11,6,7,8]
    for i,j in enumerate(mapping):raw.motor_state[j].q=float(node.policy.default_joint_pos_leg[i])
    if stand_offset:
        raw.motor_state[mapping[8]].q -= .198
    base=BaseState();base.pose.position=[0.,0.,.4];base.pose.orientation=[0.,0.,0.,1.]
    base.velocity.linear=[0.]*3;base.velocity.angular=[0.]*3
    arm=JointState();arm.name=ARM_NAMES;arm.position=[.019,1.8736,-.68316,.04965,-.99747,-.03302];arm.velocity=[0.]*6;arm.effort=[0.]*6
    arm_node=None;arm_robot=None;arm_frames=deque()
    if use_arm_executor:
        import can
        arm_interface=Path(os.environ.get('PIPER_INTERFACE_DIR',str(ROOT/'arm')))
        sys.path.insert(0,str(arm_interface))
        # Deployment uses separate processes. In this in-process ROS test,
        # both projects happen to call their configuration module "config".
        # Keep the existing Go2 module while loading the arm project's own
        # config; each imported controller retains its correct module object.
        policy_config=sys.modules.pop('config',None)
        try:
            from arm_backend import load_sdk,ArmBackend
            from run_executor import ArmExecutor
            arm_robot=load_sdk()(can_name='synthetic_no_can',can_auto_init=False)
            for name in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1','ConnectPort','DisconnectPort'):
                setattr(arm_robot,name,Mock())
            bus=Mock();bus.recv.side_effect=lambda timeout=0:arm_frames.popleft() if arm_frames else None
            arm_robot.GetCanBus=Mock(return_value=Mock(recv_bus=bus))
            backend=ArmBackend('fake',robot_factory=Mock(return_value=arm_robot))
            arm_node=ArmExecutor(backend=backend);executor.add_node(arm_node)
        finally:
            if policy_config is not None:sys.modules['config']=policy_config
        fake.destroy_publisher(arm_pub)
        arm.position=np.deg2rad(np.rint(np.rad2deg(arm.position)*1000)/1000).tolist()
    send_raw=True;send_target=False;send_arm=True
    def publish():
        if sport_remove_at[0] is not None and time.monotonic()>=sport_remove_at[0]:
            fake.destroy_publisher(sport_pub);sport_remove_at[0]=None
        if send_raw:low_pub.publish(raw)
        base_pub.publish(base)
        if send_arm:
            if use_arm_executor:
                values=np.rint(np.rad2deg(arm.position)*1000).astype(int)
                for i in range(3):
                    arm_frames.append(can.Message(arbitration_id=0x2a5+i,
                        data=struct.pack('>ii',*values[2*i:2*i+2]),is_extended_id=False,timestamp=time.time()))
            else:arm_pub.publish(arm)
        if send_target:node.publish_legs(node.policy.default_joint_pos_leg,20,1.5)
    timer=fake.create_timer(.01,publish)
    def pump(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:executor.spin_once(timeout_sec=.002)
    def until(phase,timeout):
        end=time.monotonic()+timeout
        while node.life.phase!=phase and time.monotonic()<end:
            pump(.02)
            assert node.life.phase!='FAULT',node.life.reason
        assert node.life.phase==phase,(node.life.phase,phase)
    checks=[]
    try:
        pump(2)
        assert not low_commands and not requests
        checks.append('WAIT emits no motor commands, enable calls or motion RPCs')
        assert not hasattr(node,'arm_pub') and not hasattr(node,'enable_arm')
        raw.wireless_remote=[0]*40;pump(.15)
        assert node.life.phase=='WAIT' and 'remote' not in node.receipts
        assert not requests and not low_commands
        raw.wireless_remote[0]=0x55;raw.wireless_remote[1]=0x51;pump(.1)
        assert node.life.phase=='WAIT' and 'remote' in node.receipts
        checks.append('All-zero remote in WAIT blocks readiness without latching fault; valid remote recovers readiness')

        assert node.count_publishers('/lowcmd')==2
        # An unavailable prerequisite must not latch a fault or release Sport.
        axis_packet=bytearray(raw.wireless_remote);struct.pack_into('<f',axis_packet,4,.2);raw.wireless_remote=list(axis_packet);pump(.05)
        node.events.put('prepare');pump(.1)
        assert node.life.phase=='WAIT' and not requests
        axis_packet=bytearray(raw.wireless_remote);struct.pack_into('<f',axis_packet,4,0.);raw.wireless_remote=list(axis_packet);pump(.05)
        node.events.put('prepare');until('RELEASE',2)
        release_started=node.started
        assert not low_commands and not node.hal_active
        until('STAND',3)
        assert node.hal_enable_at-node.release_confirmed_at<.1
        np.testing.assert_allclose(node.leg_start,[raw.motor_state[j].q for j in mapping],atol=1e-6)
        checks.append('Pre-release pose is seeded before RPC; HAL requested within 100 ms of release confirmation without fixed 0.6 s delay')
        if recent_lowcmd:
            assert 0 <= node.hal_enable_at-release_started<.5
            checks.append('Final Sport LowCmd immediately before release does not reject HAL activation or add a 0.5 s wait')
        until('READY',12)
        if stand_offset:
            assert .19 < max(abs(node.data.qpos[7:19]-node.policy.default_joint_pos_leg)) < .20
            assert time.monotonic()-node.started >= 5.5
            checks.append('0.198 rad standing error reaches READY after interpolation and dwell under legg 0.30 rad threshold; policy still requires manual start')
            # End the standing-error scenario before the separate policy
            # checks; fixed synthetic joints cannot respond to RL actions.
            raw.motor_state[mapping[8]].q=float(node.policy.default_joint_pos_leg[8])
            pump(.1)
        checks.append('Release/check acknowledgement precedes activation; silent LowCmd endpoints do not block takeover; failed prepare stays in WAIT')
        assert requests==[1003,1001]
        assert low_commands
        assert all(m.kp==60 and m.kd==5 for m in low_commands[-1].motor_cmd[:12])
        node.events.put('rl');until('POLICY',2);pump(.2)
        assert all(m.kp==30 and m.kd==2 for m in low_commands[-1].motor_cmd[:12])
        checks.append('Custom policy PD 30/2 reaches all 12 motors; READY retains standing PD 60/5')
        assert node.policy._observation_history.shape==(5,60)
        assert np.allclose(node.policy._observation_history[-1,-6:],np.array(arm.position)-node.policy.default_joint_pos_arm)
        checks.append('Nonzero real arm positions enter ONNX observations; release/check/stand/policy succeeds without arm enables or targets')
        if use_arm_executor:
            arm.position[0]+=.01;pump(.15)
            np.testing.assert_allclose(node.policy._observation_history[-1,-6:],
                np.array(arm.position)-node.policy.default_joint_pos_arm,atol=2e-5)
            for method in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1'):
                getattr(arm_robot,method).assert_not_called()
            checks.append('Independent executor actual CAN decoding reaches real ONNX, updates live, and sends no arm motor commands in monitor mode')
        send_arm=False;pump(.25)
        checks.append('Arm feedback loss latches leg damping; no arm commands during fault')
        assert node.life.phase=='FAULT' and not node.hal_active
        assert all(m.kp==0 and m.kd==2 for m in low_commands[-1].motor_cmd[:12])
        result={'passed':True,'checks':checks,'scope':'loopback synthetic hardware; real ONNX and managed HAL, no physical commands'}
        report='arm_readonly_integration.json'
        if stand_offset:report='arm_stand_legg_threshold_integration.json'
        if use_arm_executor:report='arm_executor_policy_integration.json'
        (ROOT/'runtime/reports').mkdir(parents=True,exist_ok=True)
        (ROOT/'runtime/reports'/report).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));return
    finally:
        if arm_node is not None:
            arm_node.close();executor.remove_node(arm_node);arm_node.destroy_node()
        hal.send_signal(signal.SIGINT);hal.wait(timeout=5);log.close()
        node.stream.close()
        for n in [node,fake]:
            if n is not None:executor.remove_node(n);n.destroy_node()
        executor.shutdown();rclpy.shutdown()


if __name__=='__main__':main()

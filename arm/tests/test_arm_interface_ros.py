from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_common
"""Loopback ROS integration, real SDK CAN parsing with mocked transmit only."""
import os
os.environ['ROS_DOMAIN_ID']='232'
os.environ['ROS_LOCALHOST_ONLY']='1'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
import json
from pathlib import Path
import struct
import time
from collections import deque
from unittest.mock import Mock
import can
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool,String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
from arm_backend import load_sdk,ArmBackend
from run_executor import ArmExecutor,PREFIX,FEEDBACK_TOPIC,TRAJECTORY_TOPIC
from keyboard_node import ArmKeyboard,joint_message


def main():
    cls=load_sdk()
    robot=cls(can_name='synthetic_no_can',can_auto_init=False)
    for name in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1','ConnectPort','DisconnectPort'):
        setattr(robot,name,Mock())
    frames=deque()
    bus=Mock();bus.recv.side_effect=lambda timeout=0:frames.popleft() if frames else None
    robot.GetCanBus=Mock(return_value=Mock(recv_bus=bus))
    backend=ArmBackend('fake',robot_factory=Mock(return_value=robot))
    rclpy.init();executor=SingleThreadedExecutor()
    arm=ArmExecutor(backend=backend);keys=ArmKeyboard();probe=Node('arm_interface_probe')
    for node in (arm,keys,probe):executor.add_node(node)
    feedback=[];statuses=[]
    probe.create_subscription(JointState,FEEDBACK_TOPIC,lambda m:feedback.append(m),1)
    probe.create_subscription(String,PREFIX+'/status',lambda m:statuses.append(json.loads(m.data)),1)
    hal=probe.create_publisher(Bool,'/locopilot/hal_active',1)
    q=np.array([0,-4.,1.5,0,0,0])
    sim=[True]
    def feed():
        if not sim[0]:return
        # After commands start, physical feedback follows the latest SDK target.
        if robot.JointCtrl.call_args is not None:
            q[:]=np.asarray(robot.JointCtrl.call_args.args)/1000.
        for ident,data in [(0x2a1,struct.pack('>6BH',1,0,1,0,0,0,0))]+[(0x2a5+i,struct.pack('>ii',*[round(v*1000) for v in q[2*i:2*i+2]])) for i in range(3)]+[(i,struct.pack('>HhBBH',240,35,30,0x40,0)) for i in range(0x261,0x267)]:
            frames.append(can.Message(arbitration_id=ident,data=data,is_extended_id=False,timestamp=time.time()))
    probe.create_timer(.01,feed)
    def pump(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:executor.spin_once(timeout_sec=.002)
    def until(predicate,seconds=3.):
        end=time.monotonic()+seconds
        while not predicate() and time.monotonic()<end:pump(.01)
        assert predicate(),(arm.last_error,backend.controller.phase)
    checks=[]
    try:
        pump(.7)
        assert feedback and np.isclose(feedback[-1].position[1],np.deg2rad(-4))
        for method in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1'):getattr(robot,method).assert_not_called()
        checks.append('Executor startup publishes actual out-of-range CAN angles without any motor transmission')
        keys.key('\n')
        until(lambda:bool(arm.last_error))
        pump(.1)
        assert not backend.failed and not backend.requested
        assert statuses[-1]['phase']=='monitor' and not statuses[-1]['ready']
        for method in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1'):
            getattr(robot,method).assert_not_called()
        checks.append('Out-of-range Enter returns to monitor with zero commands; subsequent R works without restarting either node')
        keys.key('r');until(lambda:backend.ready and keys.ready())
        assert robot.EnablePiper.call_count>=2
        assert all(abs(v)<.5 for v in q)
        assert keys.ready()
        assert statuses[-1]['target_position'] is not None
        np.testing.assert_allclose(statuses[-1]['target_position'],np.deg2rad(np.asarray(backend.controller.target)/1000.))
        checks.append('Keyboard R service performs SDK recovery and publishes real arrival feedback; fixed executor owns all CAN calls')
        previous=len(robot.JointCtrl.call_args_list)
        keys.key('a');pump(.15)
        assert len(robot.JointCtrl.call_args_list)>previous and feedback[-1].position[0]>0
        assert all(not c.args[0] > 600 for c in robot.JointCtrl.call_args_list)
        checks.append('Gamepad-style A publishes standard JointTrajectory; executor interpolates it and reports measured motion')
        second=probe.create_publisher(JointTrajectory,TRAJECTORY_TOPIC,1);pump(.2)
        second.publish(joint_message([.02,0,0,0,0,0],1.));pump(.1)
        assert '来源冲突' in arm.last_error
        probe.destroy_publisher(second);pump(.2)
        checks.append('Multiple trajectory sources rejected; source replacement is explicit')
        target=np.asarray(feedback[-1].position);target[0]+=.2
        keys.pub.publish(joint_message(target,1.));pump(.1)
        assert arm.plan is not None
        keys.cancel();pump(.15)
        assert arm.plan is None
        checks.append('P cancels queued trajectory and holds actual position')
        keys.key('\n');keys.key('b');pump(.4)
        assert all(abs(v)<.5 for v in q)
        assert not keys.homing
        assert keys.notice=='Home 完成，可直接继续运动'
        checks.append('Cartesian keyboard Home uses executor trajectory and completes at measured zero')
        hal.publish(Bool(data=True));pump(.1)
        c=probe.create_client(Trigger,PREFIX+'/recover');future=c.call_async(Trigger.Request());until(future.done)
        assert not future.result().success
        checks.append('Arm reset/recovery refused after Go2 low-level activation')
        hal.publish(Bool(data=False));pump(.1)
        executor.remove_node(keys);keys.destroy_node()
        keys=ArmKeyboard(mode='joint');executor.add_node(keys)
        until(lambda:keys.ready())
        keys.key('e');keys.key('3');keys.key('[');keys.key('j');pump(.35)
        assert abs(q[2]+.2)<.01
        assert all(abs(v)<.01 for i,v in enumerate(q) if i!=2)
        keys.key(']');keys.key('l');pump(.35)
        assert abs(q[2])<.01
        checks.append('Joint keyboard selects J3, switches 0.2/0.5 degree steps, and drives the same executor without CAN access')
        keys.key('\n');pump(.05)
        target=np.asarray(feedback[-1].position);target[0]+=.2
        keys.pub.publish(joint_message(target,1.));pump(.08)
        executor.remove_node(keys);keys.destroy_node();keys=None
        until(lambda:arm.plan is None,3.)
        checks.append('Trajectory publisher exit cancels remaining motion and holds measured position')
        sim[0]=False;pump(.7)
        assert backend.failed
        robot.MotionCtrl_1.assert_called_with(1,0,0)
        assert not arm.backend.ready
        checks.append('Arm feedback timeout latches independent arm stop')
        result={'passed':True,'checks':checks,'scope':'loopback ROS + real SDK parser; mocked CAN transport and actuation, no physical commands'}
        path=Path(__file__).resolve().parents[2]/'runtime/reports/arm_interface_integration.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        arm.close()
        for node in (keys,arm,probe):
            if node is not None:executor.remove_node(node);node.destroy_node()
        executor.shutdown();rclpy.shutdown()

if __name__=='__main__':main()

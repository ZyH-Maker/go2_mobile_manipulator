from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_common
import math
import unittest
from unittest.mock import Mock
from concurrent.futures import Future
import numpy as np
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from arm_trajectory import JointTrajectoryPlan
from keyboard_node import ArmKeyboard,joint_message
from keyboard_targets import KeyboardTargets,JointKeyboardTargets
from arm_common import ARM_NAMES


class TrajectoryTests(unittest.TestCase):
    def test_joint_keyboard_selected_axis_step_rate_and_trajectory(self):
        keys=JointKeyboardTargets();keys.selected=2;keys.step_rad=math.radians(.2)
        q=np.array([0.,.5,-.5,0.,0.,0.])
        target,duration=keys.step('j',q,q,100.)
        expected=q.copy();expected[2]-=math.radians(.2)
        np.testing.assert_allclose(target,expected)
        self.assertIsNone(keys.step('j',target,target,100.01))
        JointTrajectoryPlan(joint_message(target,duration),q,0.)

    def test_joint_keyboard_limits_and_tracking(self):
        keys=JointKeyboardTargets();keys.selected=1
        self.assertIsNone(keys.step('j',np.zeros(6),np.zeros(6),100.))
        q=np.zeros(6);q[1]=.1
        self.assertIsNone(keys.step('l',q,np.zeros(6),101.))

    def test_stop_overrides_pending_enable_callback(self):
        keyboard=Mock()
        keyboard.inflight=None;keyboard.service_generation=0;keyboard.paused=True
        enable=Future();stop=Future()
        keyboard.service_clients={'enable':Mock(),'stop':Mock()}
        keyboard.service_clients['enable'].call_async.return_value=enable
        keyboard.service_clients['stop'].call_async.return_value=stop
        ArmKeyboard.service(keyboard,'enable')
        ArmKeyboard.service(keyboard,'stop')
        keyboard.service_clients['stop'].call_async.assert_called_once()
        enable.set_result(Mock(success=True,message='accepted'))
        self.assertTrue(keyboard.paused)

    def test_real_sdk_planar_ik_solves_nearby_target_without_connection(self):
        from arm_backend import load_sdk
        load_sdk()
        from core.kinematics import PlanarKinematics
        ik=PlanarKinematics();seed=[0,60000,-80000,0,0,0]
        target=ik.step(seed,dx=.5)
        self.assertIsNotNone(target)
        delta=ik.local_position(target)-ik.local_position(seed)
        np.testing.assert_allclose(delta,[.5,0,0],atol=.03)

    def test_interpolation_and_hold_and_reordered_names(self):
        msg=joint_message([.1,0,-.1,0,0,0],1.)
        msg.joint_names=ARM_NAMES[::-1]
        msg.points[0].positions=msg.points[0].positions[::-1]
        p=JointTrajectoryPlan(msg,[0]*6,10.)
        np.testing.assert_allclose(p.sample(10.5),[.05,0,-.05,0,0,0])
        np.testing.assert_allclose(p.sample(20.),[.1,0,-.1,0,0,0])
        np.testing.assert_allclose(p.sample(9.),[0]*6)

    def test_bad_messages_and_zero_time_jumps_rejected(self):
        for kind in ('nan','range','names','velocity','time','rate','zerojump','empty'):
            msg=joint_message([0]*6,1.)
            if kind=='nan':msg.points[0].positions[0]=float('nan')
            if kind=='range':msg.points[0].positions[1]=-.2
            if kind=='names':msg.joint_names=['joint1']*6
            if kind=='velocity':msg.points[0].velocities=[0.]*6
            if kind=='time':msg.points[0].time_from_start.sec=-1
            if kind=='rate':msg.points[0].positions[0]=1.
            if kind=='zerojump':msg.points[0].positions[0]=.1;msg.points[0].time_from_start.sec=0;msg.points[0].time_from_start.nanosec=0
            if kind=='empty':msg.points=[]
            with self.subTest(kind=kind), self.assertRaises(ValueError):JointTrajectoryPlan(msg,[0]*6,0.)

    def test_duplicate_point_times_rejected(self):
        msg=joint_message([0]*6,1.);msg.points.append(msg.points[0])
        with self.assertRaises(ValueError):JointTrajectoryPlan(msg,[0]*6,0.)

    def test_keyboard_character_step_and_rate_limit(self):
        keys=KeyboardTargets();result=keys.step('a',np.zeros(6),np.zeros(6),100.)
        self.assertAlmostEqual(result[0][0],math.radians(.5))
        self.assertIsNone(keys.step('a',np.zeros(6),np.zeros(6),100.001))
        msg=joint_message(*result);JointTrajectoryPlan(msg,[0]*6,0.)

    def test_keyboard_respects_feedback_tracking_limit(self):
        keys=KeyboardTargets()
        self.assertIsNone(keys.step('a',np.array([.1,0,0,0,0,0]),np.zeros(6),100.))

    def test_keyboard_ik_failure_never_creates_target(self):
        class IK:
            def step(self,*args,**kwargs):return None
        self.assertIsNone(KeyboardTargets(IK()).step('q',np.zeros(6),np.zeros(6),100.))

    def test_home_duration_satisfies_executor_speed_contract(self):
        q=np.array([.2,.4,-.6,.1,.1,.1])
        duration=float(np.max(abs(q)))/math.radians(25)
        JointTrajectoryPlan(joint_message([0]*6,duration),q,0.)

if __name__=='__main__':unittest.main()

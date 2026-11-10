from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_common
"""Recovery adapter tests with real SDK parsing and mocked CAN/actuation."""
import unittest
from unittest.mock import Mock
from arm_backend import ArmBackend, ArmRequestRejected
import arm_config as cfg
from unittest.mock import patch
import test_state_machine as state_machine_tests


class RecoveryAdapterTests(unittest.TestCase):
    channels = state_machine_tests.KeyboardControlTests.channels
    setUp = state_machine_tests.KeyboardControlTests.setUp
    frame = state_machine_tests.KeyboardControlTests.frame
    status = state_machine_tests.KeyboardControlTests.status
    joints = state_machine_tests.KeyboardControlTests.joints
    feed_all = state_machine_tests.KeyboardControlTests.feed_all

    def adapter(self):
        self.robot.ConnectPort = Mock()
        self.robot.DisconnectPort = Mock()
        bus = Mock()
        bus.recv.return_value = None
        self.robot.GetCanBus = Mock(return_value=Mock(recv_bus=bus))
        recovery = ArmBackend('fake', robot_factory=Mock(return_value=self.robot))
        return recovery

    def test_invalid_config_rejected_before_can_factory(self):
        r=self.adapter()
        with patch.object(cfg,'RECOVERY_SPEED_PERCENT',101):
            with self.assertRaises(ValueError):r.connect()
        r.robot_factory.assert_not_called()

    def connect(self):
        recovery = self.adapter()
        recovery.request()
        self.feedback = recovery.feedback
        self.control = recovery.controller
        self.feed_all()
        return recovery

    def test_no_request_no_connection_or_commands(self):
        r = self.adapter()
        r.advance();r.close()
        r.robot_factory.assert_not_called()
        self.robot.ConnectPort.assert_not_called()
        self.robot.JointCtrl.assert_not_called()

    def test_request_only_connects_passively_and_waits_for_full_feedback(self):
        r = self.adapter();r.request();r.advance()
        self.robot.ConnectPort.assert_called_once_with(piper_init=False,start_thread=False)
        self.robot.EnablePiper.assert_not_called()
        self.robot.JointCtrl.assert_not_called()
        self.assertFalse(r.ready)
        self.now += 3.1
        with self.assertRaisesRegex(RuntimeError,'反馈不完整'):r.advance()
        self.robot.MotionCtrl_1.assert_not_called()

    def test_large_offset_refused_without_motor_commands(self):
        r = self.connect();self.q[1]=-6000;self.joints()
        with self.assertRaisesRegex(RuntimeError,'越界超过'):r.advance()
        self.robot.EnablePiper.assert_not_called()
        self.robot.ModeCtrl.assert_not_called()
        self.robot.JointCtrl.assert_not_called()
        self.assertFalse(r.ready)
        self.assertFalse(r.failed)
        self.assertFalse(r.requested)

    def test_enter_outside_range_can_retry_r_without_restart(self):
        r=self.adapter();r.request(recover=False)
        self.feedback=r.feedback;self.control=r.controller
        self.q=[2148,-3788,736,3049,25614,-346];self.feed_all()
        with self.assertRaisesRegex(ArmRequestRejected,'J2 当前 -3.788'):
            r.advance()
        self.assertFalse(r.failed)
        self.assertFalse(r.requested)
        self.assertEqual(self.control.phase,'monitor')
        for name in ('JointCtrl','ModeCtrl','EnablePiper','MotionCtrl_1'):
            getattr(self.robot,name).assert_not_called()
        r.request(recover=True);r.advance()
        for _ in range(3):
            self.now+=.11;self.feed_all();r.advance()
        self.now+=.02;self.feed_all();r.advance()
        self.assertEqual(self.control.phase,'recovering')
        self.assertEqual(self.robot.ModeCtrl.call_args.args,(1,1,5,0))
        self.q[1]=self.q[2]=0;self.now+=.02;self.feed_all();r.advance()
        self.assertTrue(r.ready)

    def test_control_fault_still_latches_and_refuses_retry(self):
        r=self.connect();r.advance()
        self.now+=.6
        with self.assertRaises(RuntimeError):r.advance()
        self.assertTrue(r.failed)
        self.robot.MotionCtrl_1.assert_called_with(1,0,0)
        with self.assertRaisesRegex(RuntimeError,'故障已锁定'):r.request()

    def test_recovery_holds_only_after_arrival_and_monitors_afterwards(self):
        r=self.connect();self.q[1]=-4000;self.q[2]=1500;self.joints()
        r.advance()
        self.assertFalse(r.ready)
        for _ in range(3):
            self.now+=.11;self.feed_all();r.advance()
        self.now+=.02;self.feed_all();r.advance()
        self.assertEqual(self.control.phase,'recovering')
        self.assertFalse(r.ready)
        self.assertEqual(self.robot.ModeCtrl.call_args.args,(1,1,5,0))
        self.assertEqual(self.robot.JointCtrl.call_args.args[1:3],(0,0))
        self.q[1]=0;self.q[2]=0;self.now+=.02;self.feed_all();r.advance()
        self.assertTrue(r.ready)
        self.now+=.6
        with self.assertRaises(RuntimeError):r.advance()
        self.assertFalse(r.ready)
        self.robot.MotionCtrl_1.assert_called_with(1,0,0)

    def test_repeated_request_does_not_restart_or_extend_deadline(self):
        r=self.connect();deadline=r.deadline
        with self.assertRaises(RuntimeError):r.request()
        self.assertEqual(r.deadline,deadline)
        self.robot.ConnectPort.assert_called_once()

    def test_stop_before_request_never_sends_to_arm(self):
        r=self.adapter();r.stop('operator');r.close()
        self.robot.MotionCtrl_1.assert_not_called()

    def test_hold_cancels_collecting_without_commands(self):
        r=self.adapter();r.request();r.hold();r.advance()
        self.assertFalse(r.requested)
        self.robot.EnablePiper.assert_not_called()
        self.robot.JointCtrl.assert_not_called()
        self.robot.MotionCtrl_1.assert_not_called()

    def test_hold_during_enable_stops_and_cannot_resume(self):
        r=self.connect();r.advance()
        self.assertTrue(self.control.owns_control)
        self.assertFalse(r.ready)
        r.hold()
        self.assertTrue(r.failed)
        self.robot.MotionCtrl_1.assert_called_with(1,0,0)
        count=self.robot.EnablePiper.call_count
        self.now+=.2;self.feed_all();r.advance()
        self.assertEqual(count,self.robot.EnablePiper.call_count)
        self.robot.JointCtrl.assert_not_called()

if __name__=='__main__':unittest.main()

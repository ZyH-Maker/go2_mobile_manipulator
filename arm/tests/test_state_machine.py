from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_common
"""Exercise motion guards with recorded-format CAN packets, never hardware."""

import itertools
import struct
import time
import unittest
from unittest.mock import Mock, patch

import can
from arm_backend import load_sdk
C_PiperInterface_V2 = load_sdk()

from arm_backend import load_sdk
load_sdk()
from core.state_machine import FEEDBACK_IDS, Feedback, PositionController as KeyboardController
import arm_config as cfg


class KeyboardControlTests(unittest.TestCase):
    channels = itertools.count()

    def setUp(self):
        self.now = 100.0
        clock = patch('core.state_machine.time.monotonic', lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.robot = C_PiperInterface_V2(
            can_name=f'keyboard_test_{next(self.channels)}', can_auto_init=False)
        for method in ('JointCtrl', 'ModeCtrl', 'EnablePiper', 'MotionCtrl_1'):
            setattr(self.robot, method, Mock())
        self.feedback = Feedback(self.robot)
        self.q = [1763, -281, -6000, 1928, 9549, 26470]
        self.feed_all()
        self.control = KeyboardController(self.robot, self.feedback)

    def frame(self, can_id, data, age=0):
        self.feedback.receive(can.Message(
            arbitration_id=can_id, data=data, is_extended_id=False,
            timestamp=time.time() - age))

    def status(self, mode=1, state=0, move=1, error=0):
        self.frame(0x2A1, struct.pack('>6BH', mode, state, move, 0, 0, 0, error))

    def joints(self):
        for i in range(3):
            self.frame(0x2A5 + i, struct.pack('>ii', *self.q[i * 2:i * 2 + 2]))

    def feed_all(self):
        self.status()
        self.joints()
        for can_id in range(0x261, 0x267):
            self.frame(can_id, struct.pack('>HhBBH', 240, 35, 30, 0x40, 0))

    def activate(self):
        self.control.take_control()
        self.finish_enable()
        self.assertEqual(self.control.phase, 'active')
        self.control.last_send = self.now
        self.robot.JointCtrl.reset_mock()

    def finish_enable(self):
        self.assertEqual(self.control.phase, 'enabling')
        self.control.tick()
        self.now += 0.11
        self.feed_all()
        self.control.tick()
        self.now += 0.11
        self.feed_all()
        self.control.tick()
        self.assertEqual(self.control.phase, 'activating')
        self.control.tick()

    def finish_reset(self):
        for _ in range(4):
            self.now += 0.31
            self.feed_all()
            self.control.tick()
        self.assertEqual(self.control.phase, 'enabling')

    def test_monitor_and_exit_never_send(self):
        self.control.tick()
        self.control.jog(1)
        self.control.close()
        for name in ('JointCtrl', 'ModeCtrl', 'EnablePiper', 'MotionCtrl_1'):
            getattr(self.robot, name).assert_not_called()

    def test_configured_joint_speed_step_and_feedback_timeout(self):
        with patch.multiple(cfg, JOINT_CONTROL_SPEED_PERCENT=30, JOINT_DEFAULT_STEP_DEG=0.3,
                            FEEDBACK_TIMEOUT_S=0.1):
            self.control = KeyboardController(self.robot, self.feedback)
            # Use short cycles to keep feedback fresh under the tighter timeout.
            self.control.take_control()
            self.finish_enable()
            initial = self.control.target[0]
            self.control.jog(1)
            self.assertEqual(self.control.target[0], initial+300)
            self.assertEqual(self.robot.ModeCtrl.call_args.args[2], 30)
            self.feedback.last_seen[0x2A5] -= 0.2
            self.control.tick()
            self.assertEqual(self.control.phase, 'stopped')

    def test_takeover_uses_actual_pose_and_waits_for_mode(self):
        self.control.take_control()
        self.finish_enable()
        for call in self.robot.JointCtrl.call_args_list:
            self.assertEqual(call.args, tuple(self.control.bounded(self.q)))
        self.assertEqual(self.robot.EnablePiper.call_count, 2)
        self.assertEqual(self.control.phase, 'active')

    def test_one_key_moves_only_selected_joint(self):
        self.activate()
        self.control.selected = 3
        self.control.jog(1)
        expected = self.control.bounded(self.q)
        expected[3] += 500
        self.robot.JointCtrl.assert_called_once_with(*expected)
        self.control.jog(1)
        self.robot.JointCtrl.assert_called_once()

    def test_boundary_offset_can_only_move_inwards(self):
        self.activate()
        self.control.selected = 1
        self.control.jog(-1)
        self.robot.JointCtrl.assert_not_called()
        self.control.jog(1)
        self.assertEqual(self.robot.JointCtrl.call_args.args[1], 500)

    def test_limit_never_exceeded(self):
        self.q[0] = 149900
        self.joints()
        self.activate()
        self.control.jog(1)
        self.assertEqual(self.robot.JointCtrl.call_args.args[0], 150000)

    def test_single_stale_joint_pair_stops_even_when_others_fresh(self):
        self.activate()
        self.feedback.last_seen[0x2A6] -= 1
        self.control.jog(1)
        self.robot.JointCtrl.assert_not_called()
        self.robot.MotionCtrl_1.assert_called_once_with(1, 0, 0)
        self.assertEqual(self.control.phase, 'stopped')

    def test_old_queued_packet_does_not_refresh_feedback(self):
        self.frame(0x2A5, struct.pack('>ii', *self.q[:2]), age=2)
        self.control.take_control()
        self.robot.JointCtrl.assert_not_called()
        self.assertIn(0x2A5, self.feedback.missing())

    def test_fault_prevents_takeover(self):
        self.status(state=7)
        self.control.take_control()
        self.robot.ModeCtrl.assert_not_called()
        self.robot.JointCtrl.assert_not_called()

    def test_motor_fault_stops_active_controller(self):
        self.activate()
        self.frame(0x263, struct.pack('>HhBBH', 240, 35, 30, 0x44, 0))
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)

    def test_enable_timeout_stops(self):
        self.control.take_control()
        self.control.enable_deadline = time.monotonic() - 1
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)

    def test_mode_change_stops(self):
        self.activate()
        self.status(mode=2)
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')

    def test_large_tracking_error_stops_without_new_target(self):
        self.activate()
        self.q[0] += 2500
        self.joints()
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.JointCtrl.assert_not_called()

    def test_stop_blocks_keys_and_resume_reads_new_actual_pose(self):
        self.activate()
        self.control.stop('test')
        self.control.jog(1)
        self.robot.JointCtrl.assert_not_called()
        self.q[0] += 1000
        self.joints()
        self.status(mode=1, state=1, move=1)
        self.control.take_control()
        self.robot.MotionCtrl_1.assert_called_with(2, 0, 0)
        self.assertEqual(self.control.target, self.control.bounded(self.q))

    def test_cleanup_stops_after_partial_takeover_failure(self):
        self.robot.JointCtrl.side_effect = RuntimeError('send failed')
        self.control.take_control()
        with self.assertRaises(RuntimeError):
            self.finish_enable()
        self.control.close()
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)

    def test_missing_feedback_cannot_enable(self):
        self.feedback.last_seen.clear()
        self.control.take_control()
        self.robot.EnablePiper.assert_not_called()
        self.robot.ModeCtrl.assert_not_called()
        self.assertEqual(set(self.feedback.missing()), set(FEEDBACK_IDS))

    def test_normal_exit_holds_actual_pose_without_fast_stop(self):
        self.activate()
        self.control.jog(1)
        self.robot.MotionCtrl_1.reset_mock()
        self.control.close()
        self.robot.JointCtrl.assert_called_with(*self.control.bounded(self.q))
        self.robot.MotionCtrl_1.assert_not_called()
        self.assertEqual(self.control.phase, 'held')

    def test_exit_with_stale_feedback_falls_back_to_fast_stop(self):
        self.activate()
        self.feedback.last_seen[0x2A5] -= 1
        self.control.close()
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)
        self.robot.JointCtrl.assert_not_called()

    def test_failed_hold_send_attempts_fast_stop(self):
        self.activate()
        self.robot.JointCtrl.side_effect = RuntimeError('send failed')
        with self.assertRaises(RuntimeError):
            self.control.close()
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)

    def test_standby_normal_status_still_gets_reset_then_two_enables(self):
        self.status(mode=0, state=0)
        self.control.take_control()
        self.robot.MotionCtrl_1.assert_called_once_with(2, 0, 0)
        self.robot.EnablePiper.assert_not_called()
        self.finish_reset()
        self.finish_enable()
        self.assertEqual(self.robot.EnablePiper.call_count, 2)
        self.assertEqual(self.control.phase, 'active')

    def test_teaching_mode_switch_sequences_stop_reset_enable(self):
        self.status(mode=2)
        self.control.take_control()
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)
        self.now += 0.21
        self.feed_all()
        self.control.tick()
        self.robot.MotionCtrl_1.assert_called_with(2, 0, 0)
        self.finish_reset()
        self.finish_enable()
        self.assertEqual(self.control.phase, 'active')

    def test_reset_resamples_pose_before_enabling(self):
        self.status(mode=0)
        self.control.take_control()
        self.q[0] += 800
        self.finish_reset()
        self.finish_enable()
        self.robot.JointCtrl.assert_called_with(*self.control.bounded(self.q))

    def test_small_out_of_range_needs_explicit_recovery_key(self):
        self.q[1] = -3075
        self.joints()
        self.control.take_control()
        self.robot.MotionCtrl_1.assert_not_called()
        self.robot.ModeCtrl.assert_not_called()
        self.control.take_control(recover_limits=True)
        self.finish_enable()
        self.assertEqual(self.control.phase, 'recovering')
        self.assertEqual(self.control.speed, 5)
        self.assertEqual(self.control.target[1], 0)
        self.assertEqual(self.control.target[0], self.q[0])
        self.assertEqual(self.control.target[2:], self.q[2:])
        self.q[1] = -200
        self.feed_all()
        self.control.tick()
        self.assertEqual(self.control.phase, 'held')

    def test_recovery_never_relaxes_target_limits(self):
        self.q[1], self.q[2] = -3075, 584
        self.joints()
        self.control.take_control(recover_limits=True)
        self.finish_enable()
        self.assertEqual(self.control.target[1:3], [0, 0])
        for call in self.robot.JointCtrl.call_args_list:
            self.assertEqual(call.args, tuple(self.control.bounded(self.q)))

    def test_large_range_error_or_motor_fault_blocks_recovery(self):
        self.q[1] = -6000
        self.joints()
        self.control.take_control(recover_limits=True)
        self.robot.ModeCtrl.assert_not_called()
        self.q[1] = -3000
        self.joints()
        self.frame(0x263, struct.pack('>HhBBH', 240, 35, 30, 0x44, 0))
        self.control.take_control(recover_limits=True)
        self.robot.ModeCtrl.assert_not_called()

    def test_active_controller_refreshes_same_target_without_keypress(self):
        self.activate()
        self.now += 0.03
        self.feed_all()
        self.control.tick()
        self.robot.JointCtrl.assert_called_once_with(*self.control.target)

    def test_enabling_does_not_send_mode_or_target_before_motors_ready(self):
        for can_id in range(0x261, 0x267):
            self.frame(can_id, struct.pack('>HhBBH', 240, 35, 30, 0, 0))
        self.control.take_control()
        self.control.tick()
        self.robot.EnablePiper.assert_called_once()
        self.robot.ModeCtrl.assert_not_called()
        self.robot.JointCtrl.assert_not_called()

    def test_reset_communication_transient_waits_without_motion(self):
        self.status(mode=0)
        self.control.take_control()
        self.status(mode=0, state=5, error=63)
        self.control.tick()
        self.assertEqual(self.control.phase, 'resetting')
        self.robot.MotionCtrl_1.assert_called_once_with(2, 0, 0)
        self.robot.EnablePiper.assert_not_called()
        self.robot.JointCtrl.assert_not_called()
        self.finish_reset()
        self.finish_enable()
        self.assertEqual(self.control.phase, 'active')

    def test_persistent_reset_communication_fault_stops(self):
        self.status(mode=0)
        self.control.take_control()
        self.now += 5.1
        self.feed_all()
        self.status(mode=0, state=5, error=63)
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)
        self.robot.JointCtrl.assert_not_called()

    def test_communication_fault_in_active_mode_stops_immediately(self):
        self.activate()
        self.status(mode=1, state=5, error=63)
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)

    def test_reset_does_not_accept_placeholder_zero_pose_immediately(self):
        self.status(mode=0)
        self.control.take_control()
        self.now += 0.4
        original = self.q.copy()
        self.q = [0] * 6
        self.feed_all()
        self.control.tick()
        self.assertEqual(self.control.phase, 'resetting')
        self.now += 0.7
        self.q = original
        self.feed_all()
        self.control.tick()
        self.assertEqual(self.control.phase, 'resetting')
        self.now += 0.31
        self.feed_all()
        self.control.tick()
        self.assertEqual(self.control.phase, 'enabling')
        self.assertEqual(self.control.target, self.control.bounded(original))

    def test_reset_waits_when_error_bits_clear_before_status(self):
        # Observed at 17:18:02 on S-V1.5-8: 5/63 -> 5/0 -> 0/0.
        self.status(mode=0)
        self.control.take_control()
        original = self.q.copy()
        for error in (63, 0):
            self.now += 0.3
            self.q = [0] * 6
            self.feed_all()
            self.status(mode=0, state=5, error=error)
            self.control.tick()
            self.assertEqual(self.control.phase, 'resetting')
            self.assertIsNone(self.control.stable_since)
            self.robot.MotionCtrl_1.assert_called_once_with(2, 0, 0)
            self.robot.EnablePiper.assert_not_called()
            self.robot.ModeCtrl.assert_not_called()
            self.robot.JointCtrl.assert_not_called()
        self.q = original
        for _ in range(2):
            self.now += 0.31
            self.feed_all()
            self.control.tick()
        self.assertEqual(self.control.phase, 'enabling')
        self.finish_enable()
        self.assertEqual(self.control.phase, 'active')
        self.robot.JointCtrl.assert_called_with(*self.control.bounded(original))

    def test_zero_error_communication_status_during_enable_waits_then_times_out(self):
        self.control.take_control()
        deadline = self.control.enable_deadline
        for elapsed in (0.2, 4.1):
            self.now = 100.0 + elapsed
            self.feed_all()
            self.status(state=5, error=0)
            self.control.tick()
            self.assertEqual(self.control.enable_deadline, deadline)
            self.robot.EnablePiper.assert_not_called()
            self.robot.ModeCtrl.assert_not_called()
            self.robot.JointCtrl.assert_not_called()
            if elapsed < 4:
                self.assertEqual(self.control.phase, 'enabling')
                self.robot.MotionCtrl_1.assert_not_called()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_once_with(1, 0, 0)

    def test_zero_error_communication_status_during_reset_times_out(self):
        self.status(mode=0)
        self.control.take_control()
        self.now += 5.1
        self.feed_all()
        self.status(mode=0, state=5, error=0)
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)
        self.robot.JointCtrl.assert_not_called()

    def test_zero_error_communication_status_in_active_mode_stops(self):
        self.activate()
        self.status(state=5, error=0)
        self.control.tick()
        self.assertEqual(self.control.phase, 'stopped')
        self.robot.MotionCtrl_1.assert_called_once_with(1, 0, 0)
        self.robot.JointCtrl.assert_not_called()

    def test_reset_transient_does_not_mask_motor_fault_or_missing_feedback(self):
        for fault in ('motor', 'missing'):
            with self.subTest(fault=fault):
                self.feed_all()
                self.control.reset()
                self.status(mode=0, state=5, error=0)
                if fault == 'motor':
                    self.frame(0x263, struct.pack('>HhBBH', 240, 35, 30, 0x44, 0))
                else:
                    self.feedback.last_seen[0x2A5] -= 1
                self.control.tick()
                self.assertEqual(self.control.phase, 'stopped')
                self.robot.MotionCtrl_1.assert_called_with(1, 0, 0)
                self.robot.JointCtrl.assert_not_called()

    def test_repeated_recovery_key_preserves_progress_and_deadlines(self):
        self.status(mode=0)
        self.control.take_control(recover_limits=True)
        deadline = self.control.reset_deadline
        for recover in (True, False, True):
            self.now += 0.1
            self.control.take_control(recover_limits=recover)
            self.assertEqual(self.control.phase, 'resetting')
            self.assertEqual(self.control.reset_deadline, deadline)
            self.assertIn('等待', self.control.notice)
            self.assertNotIn('已接管', self.control.notice)
            self.robot.MotionCtrl_1.assert_called_once_with(2, 0, 0)
            self.robot.JointCtrl.assert_not_called()


if __name__ == '__main__':
    unittest.main()

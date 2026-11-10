"""Operator-visible feedback and keyboard priorities; no CAN connection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import arm_common
import json
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from std_msgs.msg import String
from keyboard_display import screen_lines, clip_line, draw_screen
from keyboard_node import ArmKeyboard, handle_input


class KeyboardDisplayTests(unittest.TestCase):
    def node(self):
        return SimpleNamespace(mode='cartesian',status_at=10.,feedback_at=10.,
            status={'phase':'trajectory','target_position':[math.radians(12),0,0,0,0,0]},
            actual=[math.radians(10),0,0,0,0,0],homing=False,paused=False,
            generator=SimpleNamespace(selected=2,step_rad=math.radians(.2)),notice='Home 完成，可直接继续运动')

    def test_actual_and_interpolated_command_are_distinct_and_translated(self):
        lines=screen_lines(self.node(),10.01)
        text='\n'.join(lines)
        self.assertIn('轨迹执行中',text)
        self.assertNotIn('trajectory',text)
        self.assertIn('实际角度',text);self.assertIn('下发目标',text)
        row=next(row for row in lines if ' J1 ' in row)
        self.assertIn('10.000',row);self.assertIn('12.000',row)
        self.assertIn('Home 完成',text)

    def test_stale_feedback_and_missing_target_are_explicit(self):
        node=self.node();node.status.pop('target_position')
        text='\n'.join(screen_lines(node,11.))
        self.assertIn('等待执行节点',text)
        self.assertIn('当前反馈已过期',text)
        self.assertIn('--',text)

    def test_joint_selection_and_wide_character_clipping(self):
        node=self.node();node.mode='joint'
        text='\n'.join(screen_lines(node,10.01))
        self.assertIn('选中：J3',text);self.assertIn('0.2°',text)
        self.assertIn('> J3',text)
        self.assertEqual(clip_line('中A文B',4),'中A')
        self.assertEqual(clip_line('\x1b\n提示',4),'提示')

    def test_phase_changes_update_screen_without_log_spam(self):
        node=Mock();node.last_error=None;node.homing=False
        for phase in ('trajectory','held','trajectory','held'):
            ArmKeyboard.on_status(node,String(data=json.dumps({'ready':True,'phase':phase,'error':''})))
        node.get_logger.assert_not_called()
        self.assertEqual(node.last_phase,'held')

    def test_home_and_fault_notice_remain_visible(self):
        node=Mock();node.last_error='';node.homing=True;node.actual=np.zeros(6)
        ArmKeyboard.on_status(node,String(data=json.dumps({'ready':True,'phase':'held','error':''})))
        node.notify.assert_called_with('Home 完成，可直接继续运动')
        ArmKeyboard.on_status(node,String(data=json.dumps({'ready':False,'phase':'fault','error':'反馈丢失'})))
        node.notify.assert_called_with('反馈丢失')
        self.assertFalse(node.homing)

    def test_stop_exit_and_pause_override_queued_motion(self):
        node=Mock()
        self.assertTrue(handle_input(node,'aaa x'))
        node.cancel.assert_called_once_with(True);node.key.assert_not_called()
        node.reset_mock()
        self.assertFalse(handle_input(node,'aaaX'));node.key.assert_not_called()
        self.assertTrue(handle_input(node,'apd'));node.cancel.assert_called_once_with()
        node.reset_mock()
        self.assertTrue(handle_input(node,'aad'));node.key.assert_called_once_with('d')

    def test_zero_size_container_terminal_still_displays_feedback(self):
        output=io.StringIO()
        with patch('keyboard_display.shutil.get_terminal_size',return_value=(0,0)), patch('keyboard_display.sys.stdout',output):
            draw_screen(screen_lines(self.node(),10.01))
        self.assertIn('实际角度',output.getvalue())
        self.assertIn('J6',output.getvalue())


if __name__=='__main__':unittest.main()

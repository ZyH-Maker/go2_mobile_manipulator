import struct
import unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from base_core import *


class Contracts(unittest.TestCase):
    def test_standard_piper_feedback_limits(self):
        root = Path(__file__).resolve().parents[2]
        limits = arm_limits_from_urdf(root/'models/piper_limits.urdf')
        np.testing.assert_allclose(np.rad2deg(limits),
            [[-150,150],[0,180],[-170,0],[-100,100],[-70,70],[-120,120]], atol=1e-4)
        def accepted(degrees):
            q = np.deg2rad(degrees)
            return ((q >= limits[:,0]-.02) & (q <= limits[:,1]+.02)).all()
        self.assertTrue(accepted([0]*6))
        self.assertTrue(accepted([140,90,-165,0,0,0]))
        self.assertFalse(accepted([0,90,-90,0,0,130]))
        self.assertFalse(accepted([0,-4,1.5,0,0,0]))

    def test_policy_pd_bounds(self):
        self.assertEqual(validate_policy_pd(20,1.5), (20.,1.5))
        self.assertEqual(validate_policy_pd(70,0), (70.,0.))
        for kp,kd in [(0,1),(-1,1),(71,1),(20,-.1),(20,7),(float('nan'),1),(20,float('inf'))]:
            with self.assertRaises(ValueError):validate_policy_pd(kp,kd)

    def test_named_feedback(self):
        q,dq=named_arm(ARM_NAMES[::-1],list(range(6)),[0.]*6)
        np.testing.assert_equal(q,list(range(6))[::-1])
        for names,pos,vel in [(ARM_NAMES,[0]*6,[]),(ARM_NAMES,[np.nan]*6,[0]*6),
                              (['joint1']*6,[0]*6,[0]*6)]:
            with self.assertRaises((ValueError,IndexError)): named_arm(names,pos,vel)

    def test_acknowledgements_pause_and_latched_fault(self):
        s=Lifecycle();s.action('rl');self.assertEqual(s.phase,'WAIT')
        s.action('prepare',False);self.assertEqual(s.phase,'WAIT')
        for event,phase in [('prepare','RELEASE'),('lowlevel_confirmed','STAND'),
                            ('stand_ready','READY'),('rl','POLICY')]:
            s.action(event);self.assertEqual(s.phase,phase)
        s.action('pause');self.assertTrue(s.paused)
        s.action('rl',False);self.assertTrue(s.paused)
        s.action('rl');self.assertFalse(s.paused)
        s.fault('lost');s.action('prepare');self.assertEqual(s.phase,'FAULT')

    def test_remote_and_feedback_timeout(self):
        packet=bytearray(40);packet[:2]=b'\x55\x51'
        struct.pack_into('<H',packet,2,L2|R2)
        struct.pack_into('<5f',packet,4,.2,.3,0,0,.4)
        keys,axes=remote(packet);self.assertTrue(keys&L2 and keys&R2)
        np.testing.assert_allclose(axes,[.2,.4,.3])
        with self.assertRaises(ValueError):remote(bytes(40))
        check_feedback(10,[9.95])
        with self.assertRaises(ValueError):check_feedback(10,[9.8])

    def test_original_torque_clip_without_model_gate(self):
        q=np.ones(12)*4;dq=np.ones(12);torque=np.ones(12)*10
        target=clamp_leg_target(np.zeros(12),q,dq,20,1.5,torque)
        np.testing.assert_allclose(target,np.clip(np.zeros(12),q+(-torque+1.5*dq)/20,q+(torque+1.5*dq)/20))
        self.assertTrue((abs(20*(target-q)-1.5*dq)<=torque+1e-10).all())

    def test_joint_and_pd_torque_limits(self):
        q=np.zeros(12);dq=np.ones(12);torque=np.ones(12)*10
        limits=np.tile([-1,1],(12,1))
        target=clamp_leg_target(np.ones(12)*3,q,dq,20,1.5,torque,limits)
        self.assertTrue((abs(20*(target-q)-1.5*dq)<=torque+1e-10).all())
        self.assertTrue((target<=1).all())
        with self.assertRaises(ValueError):
            clamp_leg_target(q,np.ones(12)*4,dq,20,1.5,torque,limits)


if __name__=='__main__':unittest.main()

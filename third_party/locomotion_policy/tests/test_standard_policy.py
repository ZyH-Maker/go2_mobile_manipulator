"""Physical/contract regressions independent of Isaac Sim, ROS and robot hardware."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy'),str(ROOT/'scripts')]
import policy_contract as spec
from motion import Motion,trajectory_duration,quintic
from standard_runtime import StandardPolicy,observation
from validate_export import validate


class MotionTests(unittest.TestCase):
    def test_trajectory_speed_acceleration_and_endpoint_continuity(self):
        delta=torch.tensor([[2.8,-2.5,.1,1.,-.7,2.]])
        duration=trajectory_duration(delta).item()
        t=torch.linspace(0,1,20001,dtype=torch.float64)
        q=quintic(t)[:,None]*delta.double()
        velocity=torch.diff(q,dim=0)/(duration/(len(t)-1))
        acceleration=torch.diff(velocity,dim=0)/(duration/(len(t)-1))
        self.assertLessEqual(velocity.abs().max(),spec.ARM_SPEED+1e-5)
        self.assertLessEqual(acceleration.abs().max(),spec.ARM_ACCEL+1e-5)
        self.assertLess(velocity[[0,-1]].abs().max(),1e-6)
        np.testing.assert_allclose(q[-1].numpy(),delta.numpy()[0])

    def test_never_retarget_before_actual_arrival_and_stall_fault(self):
        m=Motion(1,'cpu');ids=torch.tensor([0]);m.reset(ids,False)
        m.advance(m.arm.clone(),.02,random_commands=False)
        chosen=m.target.clone()
        actual=m.start.clone()
        for _ in range(math.ceil((m.duration.item()+3.1)/.02)):
            m.advance(actual,.02,random_commands=False)
            torch.testing.assert_close(m.target,chosen)
        self.assertTrue(m.fault.item())

    def test_partial_reset_does_not_modify_other_environments(self):
        m=Motion(4,'cpu');m.reset(torch.arange(4),False)
        m.advance(m.arm.clone(),.02)
        before={key:getattr(m,key)[1:].clone() for key in ('arm','elapsed','target','commands','command_time')}
        m.reset(torch.tensor([0]),False)
        for key,value in before.items():torch.testing.assert_close(getattr(m,key)[1:],value)
        self.assertTrue(torch.all(m.commands[0]==0))

    def test_command_ramps_and_hold_arm_mode(self):
        m=Motion(1,'cpu');m.reset(torch.tensor([0]),False)
        initial=m.arm.clone();m.set_commands(torch.tensor([spec.COMMAND_HIGH]))
        m.advance(initial,.02,random_commands=False,update_arm=False)
        self.assertTrue(torch.all(m.commands.abs()<=m.rate*.02+1e-7))
        torch.testing.assert_close(m.arm,initial)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(spec.MODEL.parent/'scene_standard_flat.xml'))

    def test_model_has_standard_arm_and_only_18_active_joints(self):
        m=self.model
        self.assertEqual(m.nv,24)
        self.assertEqual(m.nu,18)
        self.assertAlmostEqual(m.body('piper_base').mass[0],1.02)
        np.testing.assert_allclose(m.body('link3').pos,[.28503,0,0])
        np.testing.assert_allclose(m.joint('joint6').range,[-math.radians(120),math.radians(120)])
        self.assertTrue(np.all(m.body_inertia[1:]>0))

    def test_standard_urdf_forward_kinematics_matches_mujoco(self):
        urdf=ET.parse(ROOT.parents[1]/'models/piper/piper.urdf').getroot()
        data=mujoco.MjData(self.model)
        data.qpos[:]=0
        q=np.array([.3,1.2,-1.8,.4,-.2,.8])
        data.qpos[3]=1
        for name,v in zip(spec.ARMS,q):data.qpos[self.model.joint(name).qposadr]=v
        mujoco.mj_forward(self.model,data)
        x=np.eye(4);x[2,3]=.07
        for name,v in zip(spec.ARMS,q):
            joint=urdf.find(f"joint[@name='{name}']");origin=joint.find('origin')
            r,p,y=map(float,origin.get('rpy').split())
            rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
            ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
            rz=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
            t=np.eye(4);t[:3,:3]=rz@ry@rx;t[:3,3]=list(map(float,origin.get('xyz').split()))
            turn=np.eye(4);turn[:2,:2]=[[np.cos(v),-np.sin(v)],[np.sin(v),np.cos(v)]]
            x=x@t@turn
        np.testing.assert_allclose(data.xpos[self.model.body('link6').id],x[:3,3],atol=1e-7)
        np.testing.assert_allclose(data.xmat[self.model.body('link6').id].reshape(3,3),x[:3,:3],atol=1e-7)

    def test_motion_graph_is_connected_and_all_paths_clear(self):
        bank=json.loads((ROOT/'config/arm_motion_library.json').read_text())
        self.assertEqual(bank['model_sha256'],hashlib.sha256(spec.MODEL.read_bytes()).hexdigest())
        visited={0};todo=[0]
        while todo:
            for n in bank['edges'][todo.pop()]:
                if n not in visited:visited.add(n);todo.append(n)
        self.assertEqual(len(visited),len(bank['poses']))
        m=self.model;d=mujoco.MjData(m)
        arm_ids={m.body(n).id for n in ('piper_base','gripper_base',*(f'link{i}' for i in range(1,9)))}
        for i,targets in enumerate(bank['edges']):
            for j in targets:
                if j<i:continue
                for t in np.linspace(0,1,bank['path_samples']):
                    d.qpos[:]=0;d.qpos[2]=.22;d.qpos[3]=1
                    for n,v in zip(spec.LEGS,spec.HOME_LEGS):d.qpos[m.joint(n).qposadr]=v
                    for n,a,b in zip(spec.ARMS,bank['poses'][i],bank['poses'][j]):d.qpos[m.joint(n).qposadr]=a*(1-t)+b*t
                    mujoco.mj_forward(m,d)
                    for c in d.contact:
                        bodies={int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}
                        self.assertFalse(c.dist<0 and bodies & arm_ids,(i,j,t,bodies))


class ExportTests(unittest.TestCase):
    def test_exported_actor_parity_history_and_schema_rejection(self):
        torch.manual_seed(42)
        actor=torch.nn.Sequential(torch.nn.Linear(300,32),torch.nn.ELU(),torch.nn.Linear(32,12)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);spec.write_manifest(path)
            torch.jit.trace(actor,torch.zeros(1,300)).save(str(path/'policy.pt'))
            torch.onnx.export(actor,torch.zeros(1,300),str(path/'policy.onnx'),
                input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
            result=validate(path,8);self.assertTrue(result['passed'])
            runner=StandardPolicy(path)
            runner.infer(np.ones(60));runner.infer(np.full(60,2))
            np.testing.assert_array_equal(runner.history[-2:],np.array([[1]*60,[2]*60]))
            manifest=json.loads((path/'policy_contract.json').read_text());manifest['schema']='legacy'
            (path/'policy_contract.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):StandardPolicy(path)


if __name__=='__main__':unittest.main()

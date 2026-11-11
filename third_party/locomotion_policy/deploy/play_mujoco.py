"""Standard PiPER sim2sim: explicit policy bundle, flat/rough, five commands, JSON metrics."""
import argparse
import json
from pathlib import Path
import sys
import time
import math
import numpy as np
import mujoco
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import policy_contract as spec
from motion import Motion
from standard_runtime import StandardPolicy, observation


def rollout(args, commands, case):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = mujoco.MjModel.from_xml_path(str(spec.MODEL.parent/f'scene_standard_{args.terrain}.xml'))
    data = mujoco.MjData(model)
    legs = [model.joint(n).qposadr[0] for n in spec.LEGS]
    leg_v = [model.joint(n).dofadr[0] for n in spec.LEGS]
    arms = [model.joint(n).qposadr[0] for n in spec.ARMS]
    arm_v = [model.joint(n).dofadr[0] for n in spec.ARMS]
    leg_act = [model.actuator(n).id for n in spec.LEGS]
    arm_act = [model.actuator(n).id for n in spec.ARMS]
    arm_body = {model.body(n).id for n in ('piper_base','gripper_base',*(f'link{i}' for i in range(1,9)))}
    foot_body = {model.body(n+'_foot').id for n in ('FL','FR','RL','RR')}
    payload_id = model.body('payload').id
    model.body_inertia[payload_id] *= args.payload/model.body_mass[payload_id]
    model.body_mass[payload_id] = args.payload
    mujoco.mj_setConst(model,data)
    data.qpos[2] = .32; data.qpos[3] = 1.
    data.qpos[legs] = spec.HOME_LEGS; data.qpos[arms] = spec.HOME_ARM
    mujoco.mj_forward(model,data)
    motion = Motion(1,'cpu'); motion.reset(torch.tensor([0]),random_arm=False)
    policy = None if args.model_only else StandardPolicy(args.policy)
    phase = np.array([0.,.5,.5,0.])
    metrics = []; failure = None; collision_pairs = []
    arm_error_time = 0.
    viewer = None
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model,data)
    previous_target = np.array(spec.HOME_LEGS)
    try:
        for step in range(round(args.seconds/spec.CONTROL_DT)):
            start = time.monotonic()
            rotation = data.xmat[model.body('base').id].reshape(3,3)
            clocks = phase if motion.commands[0,:3].norm()>.01 else np.full(4,-1.)
            previous = np.zeros(12) if policy is None else policy.previous
            frame = observation(rotation.T@data.qvel[:3],data.qvel[3:6],rotation.T@np.array([0.,0.,-1.]),
                motion.commands[0].numpy(),data.qpos[legs],data.qvel[leg_v],previous,clocks,data.qpos[arms])
            target = np.array(spec.HOME_LEGS) if policy is None else policy.infer(frame)
            # First settle at zero commands, then ramp to the requested case.
            requested = np.zeros(5) if data.time<args.settle else np.array(commands)
            motion.set_commands(torch.tensor(requested,dtype=torch.float32).reshape(1,5))
            motion.advance(torch.tensor(data.qpos[arms],dtype=torch.float32).reshape(1,6),spec.CONTROL_DT,
                random_commands=False,update_arm=not args.hold_arm)
            phase = (phase+1.4*spec.CONTROL_DT)%1.
            for sub in range(spec.DECIMATION):
                delayed = previous_target if args.motor_delay and sub==0 else target
                kp,kd = (60.,5.) if args.model_only else (spec.LEG_KP,spec.LEG_KD)
                leg_tau = kp*(delayed-data.qpos[legs])-kd*data.qvel[leg_v]
                data.ctrl[leg_act] = np.clip(leg_tau,model.actuator_ctrlrange[leg_act,0],model.actuator_ctrlrange[leg_act,1])
                data.ctrl[arm_act] = motion.arm[0].numpy()
                mujoco.mj_step(model,data)
            previous_target = target.copy()
            rotation = data.xmat[model.body('base').id].reshape(3,3)
            pitch = math.asin(np.clip(-rotation[2,0],-1,1))
            roll = math.atan2(rotation[2,1],rotation[2,2])
            ground_hits = []
            for x in np.linspace(-.3,.3,7):
                for y in np.linspace(-.3,.3,7):
                    yaw = math.atan2(rotation[1,0],rotation[0,0])
                    point = data.qpos[:3]+np.array([math.cos(yaw)*x-math.sin(yaw)*y,
                        math.sin(yaw)*x+math.cos(yaw)*y,1.])
                    geomid = np.array([-1],dtype=np.int32)
                    # Group 0 is terrain only; all robot collisions are group 3.
                    ray = mujoco.mj_ray(model,data,point,np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,geomid)
                    if ray>=0: ground_hits.append(point[2]-ray)
            ground = float(np.mean(ground_hits)) if ground_hits else float('nan')
            contacts = False
            for contact_id, contact in enumerate(data.contact):
                a,b = int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
                force = np.zeros(6); mujoco.mj_contactForce(model,data,contact_id,force)
                if np.linalg.norm(force[:3])>5.:
                    if a in arm_body or b in arm_body or (a==0 and b not in foot_body) or (b==0 and a not in foot_body):
                        contacts = True
                        collision_pairs.append([model.body(a).name,model.body(b).name])
            command = motion.commands[0].numpy()
            if data.time>args.settle+2.:
                velocity = rotation.T@data.qvel[:3]
                metrics.append([np.linalg.norm(velocity[:2]-command[:2]),abs(data.qvel[5]-command[2]),
                    abs(pitch-command[3]),abs(data.qpos[2]-ground-spec.BASE_HEIGHT-command[4]),
                    np.max(np.abs(data.qpos[arms]-motion.arm[0].numpy()))])
            arm_error_time = arm_error_time+spec.CONTROL_DT if np.max(np.abs(data.qpos[arms]-motion.arm[0].numpy()))>.12 else 0.
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(ground):failure='nonfinite'
            elif abs(roll)>.8 or abs(pitch)>.8 or data.qpos[2]-ground<.08:failure='fall'
            elif contacts:failure='undesired_contact'
            elif motion.fault.item():failure='arm_arrival_timeout'
            elif arm_error_time>.5:failure='arm_tracking_error'
            if failure:break
            if viewer:
                if not viewer.is_running():break
                viewer.sync()
            if args.realtime or viewer:
                time.sleep(max(0.,spec.CONTROL_DT-(time.monotonic()-start)))
    finally:
        if viewer:viewer.close()
    values = np.array(metrics)
    rms = np.sqrt(np.mean(values**2,axis=0)).tolist() if len(values) else None
    passed = failure is None and rms is not None and all(v<t for v,t in zip(rms,[.25,.2,.12,.04,.06]))
    if args.model_only:
        passed = failure is None and rms is not None
    return dict(case=case,terrain=args.terrain,commands=commands,seed=args.seed,payload_kg=args.payload,
        simulated_seconds=float(data.time),failure=failure,collision_pairs=collision_pairs,
        rmse=dict(zip(['velocity_m_s','yaw_rad_s','pitch_rad','height_m','arm_rad'],rms)) if rms else None,
        passed=passed,kind='model_pd_smoke' if args.model_only else 'policy_sim2sim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,help='Export directory containing ONNX and policy_contract.json')
    p.add_argument('--terrain',choices=['flat','rough'],default='flat')
    p.add_argument('--commands',type=float,nargs=5,default=[0.,0.,0.,0.,0.],metavar=('VX','VY','WZ','PITCH','DH'))
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--settle',type=float,default=2.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--payload',type=float,default=.1)
    p.add_argument('--motor-delay',type=int,choices=[0,1],default=1)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--realtime',action='store_true')
    p.add_argument('--hold-arm',action='store_true')
    p.add_argument('--model-only',action='store_true',help='PD smoke only; does not validate any learned policy')
    p.add_argument('--suite',action='store_true')
    p.add_argument('--report',type=Path,default=ROOT/'reports/sim2sim.json')
    args=p.parse_args()
    if not args.model_only and not args.policy:p.error('--policy required unless --model-only')
    if args.seconds<=args.settle+2 or args.payload<=0:p.error('seconds must exceed settle+2; payload must be positive')
    if not np.isfinite(args.commands).all() or np.any(np.array(args.commands)<spec.COMMAND_LOW) or np.any(np.array(args.commands)>spec.COMMAND_HIGH):
        p.error('commands outside policy training ranges')
    cases=[('custom',args.commands)] if not args.suite else [
        ('stand',[0,0,0,0,0]),('pitch_forward',[0,0,0,.2,0]),('pitch_back',[0,0,0,-.2,0]),
        ('lower',[0,0,0,0,-.06]),('pose_combined',[0,0,0,.15,-.05]),
        ('forward',[.2,0,0,0,0]),('backward',[-.15,0,0,0,0]),
        ('lateral',[0,.1,0,0,0]),('turn',[0,0,.2,0,0]),('walk_pose',[.1,0,.1,.1,-.03])]
    results=[rollout(args,commands,name) for name,commands in cases]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2,allow_nan=False))
    return 0 if all(x['passed'] for x in results) else 1


if __name__=='__main__':
    raise SystemExit(main())

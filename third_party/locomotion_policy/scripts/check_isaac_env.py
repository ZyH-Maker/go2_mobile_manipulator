"""Isaac physics smoke: asset/ABI/reset/observation purity, before spending PPO time."""
import argparse
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',choices=['Go2-Piper-Standard-Flat-v0','Go2-Piper-Standard-Rough-v0'],required=True)
p.add_argument('--num-envs',type=int,default=8)
p.add_argument('--steps',type=int,default=200)
p.add_argument('--report',type=Path,default=Path('reports/isaac_smoke.json'))
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
app=AppLauncher(args)
try:
    import gymnasium as gym
    import torch
    import locomanipulation_teleop_isaaclab.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    cfg=parse_env_cfg(args.task,device=args.device,num_envs=args.num_envs)
    env=gym.make(args.task,cfg=cfg)
    obs,_=env.reset()
    raw=env.unwrapped
    assert obs['policy'].shape==(args.num_envs,300)
    before={name:getattr(raw.motion,name).clone() for name in ('arm','elapsed','commands')}
    first=raw._get_observations()['policy'].clone()
    second=raw._get_observations()['policy'].clone()
    torch.testing.assert_close(first,second)
    for name,value in before.items():torch.testing.assert_close(getattr(raw.motion,name),value)
    failures=0
    for _ in range(args.steps):
        obs,reward,terminated,truncated,_=env.step(torch.zeros((args.num_envs,12),device=raw.device))
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        failures+=int(terminated.sum().item())
    if args.num_envs>1:
        before=raw.motion.commands[1:].clone()
        raw._reset_idx(torch.tensor([0],device=raw.device))
        torch.testing.assert_close(raw.motion.commands[1:],before)
    report=dict(task=args.task,num_envs=args.num_envs,steps=args.steps,
        active_joints=raw._robot.joint_names,zero_action_terminations=failures,
        passed=True,note='Physics/interface smoke only; zero actions are not a trained locomotion policy')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    env.close()
finally:
    app.app.close()

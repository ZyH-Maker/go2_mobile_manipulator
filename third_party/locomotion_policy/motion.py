"""Batched, C2-continuous arm trajectories and rate-limited five-axis commands."""
import hashlib
import json
import math
import torch
import policy_contract as spec


def quintic(t):
    t = t.clamp(0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def trajectory_duration(delta, speed=spec.ARM_SPEED, acceleration=spec.ARM_ACCEL):
    """Exact peak derivatives of quintic smoothstep, for rest-to-rest segments."""
    distance = delta.abs().amax(dim=-1)
    return torch.maximum(1.875*distance/speed, (5.773503*distance/acceleration).sqrt()).clamp_min(2.5)


class Motion:
    def __init__(self, count, device, library=None):
        library = library or spec.ROOT/"config/arm_motion_library.json"
        bank = json.loads(library.read_text())
        if bank["model_sha256"] != hashlib.sha256(spec.MODEL.read_bytes()).hexdigest():
            raise ValueError("Arm motion library is stale: rebuild after model changes")
        self.poses = torch.tensor(bank["poses"], device=device)
        width = max(map(len, bank["edges"]))
        self.edges = torch.zeros((len(self.poses), width), device=device, dtype=torch.long)
        self.degree = torch.tensor([len(x) for x in bank["edges"]], device=device)
        for i, targets in enumerate(bank["edges"]):
            self.edges[i, :len(targets)] = torch.tensor(targets, device=device)
        self.node = torch.zeros(count, device=device, dtype=torch.long)
        self.target_node = self.node.clone()
        self.start = self.poses[self.node].clone()
        self.target = self.start.clone()
        self.arm = self.start.clone()
        self.elapsed = torch.zeros(count, device=device)
        self.duration = torch.ones(count, device=device)
        self.dwell = torch.zeros(count, device=device)
        self.arrived = torch.zeros(count, device=device, dtype=torch.bool)
        self.fault = self.arrived.clone()
        self.commands = torch.zeros((count,5), device=device)
        self.command_target = self.commands.clone()
        self.command_time = torch.zeros(count, device=device)
        self.low = torch.tensor(spec.COMMAND_LOW, device=device)
        self.high = torch.tensor(spec.COMMAND_HIGH, device=device)
        self.rate = torch.tensor(spec.COMMAND_RATE, device=device)

    def reset(self, ids, random_arm=True):
        nodes = torch.randint(len(self.poses), (len(ids),), device=self.node.device) if random_arm else torch.zeros_like(ids)
        self.node[ids] = nodes
        self.target_node[ids] = nodes
        self.arm[ids] = self.poses[nodes]
        self.start[ids] = self.arm[ids]
        self.target[ids] = self.arm[ids]
        self.elapsed[ids] = 0
        self.duration[ids] = 1
        self.dwell[ids] = 0
        self.arrived[ids] = True
        self.fault[ids] = False
        self.commands[ids] = 0
        self.command_target[ids] = 0
        self.command_time[ids] = 0

    def set_commands(self, commands):
        if commands.shape != self.commands.shape or not torch.isfinite(commands).all():
            raise ValueError("Expected finite (N,5) commands")
        self.command_target[:] = commands.clamp(self.low, self.high)

    def advance(self, actual_arm, dt, curriculum=1., random_commands=True, update_arm=True):
        if not dt > 0:
            raise ValueError("dt must be positive")
        if random_commands:
            self.command_time -= dt
            ids = torch.where(self.command_time <= 0)[0]
            if len(ids):
                target = self.low + torch.rand((len(ids),5), device=self.arm.device)*(self.high-self.low)
                target *= curriculum
                # Stationary and moving pose tracking are both sampled throughout episodes.
                target[torch.rand(len(ids), device=self.arm.device)<.35, :3] = 0
                self.command_target[ids] = target
                self.command_time[ids] = 3.+3.*torch.rand(len(ids), device=self.arm.device)
        self.commands += (self.command_target-self.commands).clamp(-self.rate*dt, self.rate*dt)
        if not update_arm:
            return
        moving = ~self.arrived
        self.elapsed[moving] += dt
        blend = quintic(self.elapsed/self.duration).unsqueeze(-1)
        self.arm[:] = self.start + blend*(self.target-self.start)
        reached = (self.elapsed >= self.duration) & ((actual_arm-self.target).abs().amax(-1) < math.radians(2.))
        self.node[reached] = self.target_node[reached]
        self.arrived |= reached
        self.fault |= moving & (self.elapsed > self.duration+3.)
        self.dwell[self.arrived] -= dt
        ids = torch.where(self.arrived & (self.dwell<=0) & ~self.fault)[0]
        if len(ids):
            choice = (torch.rand(len(ids), device=self.arm.device)*self.degree[self.node[ids]]).long()
            targets = self.edges[self.node[ids], choice]
            self.target_node[ids] = targets
            self.start[ids] = self.arm[ids]
            self.target[ids] = self.poses[targets]
            self.duration[ids] = trajectory_duration(self.target[ids]-self.start[ids])*(1.+.5*torch.rand(len(ids),device=self.arm.device))
            self.elapsed[ids] = 0
            self.arrived[ids] = False
            # Mixture of continuous segments and stop-and-hold segments. Always finish first.
            self.dwell[ids] = torch.where(torch.rand(len(ids),device=self.arm.device)<.5,
                torch.zeros(len(ids),device=self.arm.device), .3+1.2*torch.rand(len(ids),device=self.arm.device))

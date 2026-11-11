# Derived from the upstream BSD-3-Clause DirectRLEnv; see ../../../../../LICENSE.
"""Leg-only PPO with moving standard PiPER and commanded speed/pitch/height."""
import hashlib
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import math as math_utils
import policy_contract as spec
from motion import Motion
from .locomanipulation_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from ...assets.go2_piper_standard_asset import USD


class LocomotionManipulationEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if abs(cfg.sim.dt-spec.DT)>1.e-10 or cfg.decimation!=spec.DECIMATION:
            raise ValueError('Control timing must match policy_contract.py; update the shared contract before changing it')
        fingerprint = USD.parent/'model.sha256'
        if not USD.exists() or not fingerprint.exists() or fingerprint.read_text().strip() != hashlib.sha256(spec.MODEL.read_bytes()).hexdigest():
            raise RuntimeError('Missing/stale USD. Run scripts/assets/convert_usd.py --headless first.')
        super().__init__(cfg, render_mode, **kwargs)
        self.leg_ids = self._robot.find_joints(list(spec.LEGS), preserve_order=True)[0]
        self.arm_ids = self._robot.find_joints(list(spec.ARMS), preserve_order=True)[0]
        if self._robot.num_joints != 18:
            raise RuntimeError(f'Expected 18 active joints, got {self._robot.joint_names}; inspect USD conversion')
        self.feet_ids = self._robot.find_bodies('.*_foot')[0]
        self.feet_sensor_ids = self._contact.find_bodies('.*_foot')[0]
        self.arm_sensor_ids = self._contact.find_bodies('link[1-8]|gripper_base|piper_base')[0]
        self.body_sensor_ids = self._contact.find_bodies('base|.*_hip|.*_thigh')[0]
        if len(self.feet_ids) != 4 or len(self.arm_sensor_ids) < 6:
            raise RuntimeError('Missing feet/arm bodies after MJCF conversion')
        self.motion = Motion(self.num_envs, self.device)
        self.actions = torch.zeros((self.num_envs,12),device=self.device)
        self.previous_actions = self.actions.clone()
        self.older_actions = self.actions.clone()
        self.leg_target = self.actions.clone()
        self.default_leg = torch.tensor(spec.HOME_LEGS, device=self.device)
        self.default_arm = torch.tensor(spec.HOME_ARM, device=self.device)
        self.phase = torch.zeros((self.num_envs,4),device=self.device)
        self.phase_offset = torch.tensor((0.,.5,.5,0.),device=self.device)
        self.history = torch.zeros((self.num_envs,spec.HISTORY,spec.FRAME_SIZE),device=self.device)
        self.history_version = torch.full((self.num_envs,), -1,device=self.device,dtype=torch.long)
        self.arm_error_time = torch.zeros(self.num_envs,device=self.device)
        self.velocity_distance = torch.zeros(self.num_envs,device=self.device)
        self.episode_metrics = {}
        self._initialized = False

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations['robot'] = self._robot
        self._contact = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors['contact'] = self._contact
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors['height_scanner'] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # No broad arm-ground / arm-robot collision suppression.
        light = sim_utils.DomeLightCfg(intensity=1500.)
        light.func('/World/Light',light)

    def set_commands(self, commands):
        """External high-level/evaluation commands, shape (N,5); disable random_commands."""
        self.motion.set_commands(commands)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError('Non-finite policy action')
        self.older_actions[:] = self.previous_actions
        self.previous_actions[:] = self.actions
        self.actions[:] = actions.clamp(-spec.ACTION_CLIP,spec.ACTION_CLIP)
        self.leg_target[:] = self.default_leg + spec.ACTION_SCALE*(
            spec.ACTION_ALPHA*self.actions+(1-spec.ACTION_ALPHA)*self.previous_actions)
        curriculum = min(1., .25+.75*self.common_step_counter/max(1,self.cfg.command_curriculum_steps))
        self.motion.advance(self._robot.data.joint_pos[:,self.arm_ids],self.step_dt,
            curriculum=curriculum,random_commands=self.cfg.random_commands,update_arm=self.cfg.arm_motion)
        self.phase[:] = (self.phase+1.4*self.step_dt)%1.
        self.velocity_distance += self.motion.commands[:,:2].norm(dim=-1)*self.step_dt
        self.arm_error_time = torch.where(
            (self._robot.data.joint_pos[:,self.arm_ids]-self.motion.arm).abs().amax(-1)>self.cfg.arm_tracking_limit,
            self.arm_error_time+self.step_dt,0.)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.leg_target,joint_ids=self.leg_ids)
        self._robot.set_joint_position_target(self.motion.arm,joint_ids=self.arm_ids)

    def _get_observations(self):
        moving = self.motion.commands[:,:3].norm(dim=-1) > .01
        clocks = torch.where(moving[:,None],self.phase,-torch.ones_like(self.phase))
        frame = torch.cat((self._robot.data.root_lin_vel_b,self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,self.motion.commands,
            self._robot.data.joint_pos[:,self.leg_ids]-self.default_leg,
            self._robot.data.joint_vel[:,self.leg_ids],self.actions,clocks,
            self._robot.data.joint_pos[:,self.arm_ids]-self.default_arm),dim=-1)
        if frame.shape[-1] != spec.FRAME_SIZE:
            raise RuntimeError('Policy ABI observation size mismatch')
        # Repeated observation reads do not advance motion, phase, history or noise.
        fresh = self.history_version != self.common_step_counter
        if self.cfg.noise and fresh.any():
            scale = torch.tensor([.04]*3+[.03]*3+[.01]*3+[0.]*5+[.01]*12+
                                 [.1]*12+[0.]*12+[0.]*4+[.005]*6,device=self.device)
            frame[fresh] += torch.randn_like(frame[fresh])*scale
        self.history[fresh] = torch.cat((self.history[fresh,1:],frame[fresh,None]),dim=1)
        self.history_version[fresh] = self.common_step_counter
        return {'policy': self.history.flatten(start_dim=1)}

    def ground_height(self):
        heights = self._height_scanner.data.ray_hits_w[...,2]
        valid = torch.isfinite(heights)
        return torch.where(valid,heights,0.).sum(-1)/valid.sum(-1).clamp_min(1)

    def metrics(self):
        roll,pitch,_ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        roll = torch.atan2(roll.sin(),roll.cos())
        pitch = torch.atan2(pitch.sin(),pitch.cos())
        forces = self._contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        arm_contact = forces[:,self.arm_sensor_ids].amax(-1)>self.cfg.termination_contact_force
        body_contact = forces[:,self.body_sensor_ids].amax(-1)>self.cfg.termination_contact_force
        return dict(velocity_error=(self._robot.data.root_lin_vel_b[:,:2]-self.motion.commands[:,:2]).norm(dim=-1),
            yaw_error=(self._robot.data.root_ang_vel_b[:,2]-self.motion.commands[:,2]).abs(),
            pitch_error=(pitch-self.motion.commands[:,3]).abs(),roll=roll,pitch=pitch,
            height_error=(self._robot.data.root_pos_w[:,2]-self.ground_height()-spec.BASE_HEIGHT-self.motion.commands[:,4]).abs(),
            arm_contact=arm_contact,body_contact=body_contact,
            arm_tracking_error=(self._robot.data.joint_pos[:,self.arm_ids]-self.motion.arm).abs().amax(-1))

    def _get_rewards(self):
        m = self.metrics()
        moving = self.motion.commands[:,:3].norm(dim=-1)>.01
        foot_contact = self._contact.data.net_forces_w[:,self.feet_sensor_ids].norm(dim=-1)>5.
        stance = (self.phase < .65) | ~moving[:,None]
        contact_match = (foot_contact==stance).float().mean(-1)
        foot_height = self._robot.data.body_pos_w[:,self.feet_ids,2]-self.ground_height()[:,None]
        swing = (~stance).float()
        clearance = ((foot_height-.06).square()*swing).mean(-1)
        dq = self._robot.data.joint_vel[:,self.leg_ids]
        tau = self._robot.data.applied_torque[:,self.leg_ids]
        terms = dict(velocity=2.*torch.exp(-m['velocity_error'].square()/.10),
            yaw=.5*torch.exp(-m['yaw_error'].square()/.25),
            pitch=1.5*torch.exp(-m['pitch_error'].square()/.015),
            height=1.5*torch.exp(-m['height_error'].square()/.0025),
            roll=-2.*m['roll'].square(), vertical_velocity=-1.*self._robot.data.root_lin_vel_b[:,2].square(),
            angular_velocity=-.1*self._robot.data.root_ang_vel_b[:,:2].square().sum(-1),
            contacts=-2.*(m['arm_contact'].float()+m['body_contact'].float()),
            gait=.5*contact_match,clearance=-5.*clearance,
            torque=-2.5e-6*tau.square().sum(-1),energy=-1.e-4*(tau*dq).abs().sum(-1),
            action_rate=-.01*(self.actions-self.previous_actions).square().sum(-1),
            action_smoothness=-.001*(self.actions-2*self.previous_actions+self.older_actions).square().sum(-1))
        if not self.episode_metrics:
            self.episode_metrics = {k:torch.zeros(self.num_envs,device=self.device) for k in terms}
        for key,value in terms.items():
            self.episode_metrics[key] += value*self.step_dt
        reward = sum(terms.values())*self.step_dt
        if not torch.isfinite(reward).all():
            raise RuntimeError('Non-finite reward; inspect simulator/model instead of hiding failure')
        return reward

    def _get_dones(self):
        m = self.metrics()
        clearance = self._robot.data.root_pos_w[:,2]-self.ground_height()
        fallen = (m['roll'].abs()>.8)|(m['pitch'].abs()>.8)|(clearance<self.cfg.min_ground_clearance)
        finite = torch.isfinite(self._robot.data.root_state_w).all(-1) & torch.isfinite(self._robot.data.joint_pos).all(-1)
        terminated = fallen|m['arm_contact']|m['body_contact']|self.motion.fault|(self.arm_error_time>.5)|~finite
        return terminated, self.episode_length_buf>=self.max_episode_length-1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # Curriculum: move up only after commanded travel; never reward standing for progression.
        if self._initialized and self.cfg.terrain.terrain_type=='generator':
            distance = (self._robot.data.root_pos_w[env_ids,:2]-self._terrain.env_origins[env_ids,:2]).norm(dim=-1)
            completed = self.episode_length_buf[env_ids]>=self.max_episode_length-1
            up = completed & (distance>2.) & (distance>.7*self.velocity_distance[env_ids])
            down = ~completed
            self._terrain.update_env_origins(env_ids,up,down)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.motion.reset(env_ids,random_arm=self.cfg.random_arm)
        q = self._robot.data.default_joint_pos[env_ids].clone()
        dq = torch.zeros_like(q)
        q[:,self.arm_ids] = self.motion.arm[env_ids]
        q[:,self.leg_ids] = self.default_leg
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:,:3] += self._terrain.env_origins[env_ids]
        root[:,7:] = 0
        self._robot.write_root_pose_to_sim(root[:,:7],env_ids)
        self._robot.write_root_velocity_to_sim(root[:,7:],env_ids)
        self._robot.write_joint_state_to_sim(q,dq,env_ids=env_ids)
        self._robot.set_joint_position_target(q,env_ids=env_ids)
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.older_actions[env_ids] = 0
        self.leg_target[env_ids] = self.default_leg
        self.phase[env_ids] = self.phase_offset
        self.history[env_ids] = 0
        self.history_version[env_ids] = -1
        self.arm_error_time[env_ids] = 0
        self.velocity_distance[env_ids] = 0
        self.extras['log'] = {}
        for key,value in self.episode_metrics.items():
            self.extras['log']['Reward/'+key] = value[env_ids].mean()/self.max_episode_length_s
            value[env_ids] = 0
        if self.cfg.terrain.terrain_type=='generator':
            self.extras['log']['Curriculum/terrain_level'] = self._terrain.terrain_levels.float().mean()
        self._initialized = True

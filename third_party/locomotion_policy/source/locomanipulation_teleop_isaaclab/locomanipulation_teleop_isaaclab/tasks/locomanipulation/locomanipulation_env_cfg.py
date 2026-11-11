"""Only Flat/Rough blind tasks, sharing a standard PiPER robot and policy ABI."""
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from locomanipulation_teleop_isaaclab.assets.go2_piper_standard_asset import GO2_CFG
import policy_contract as spec


@configclass
class EventCfg:
    material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot",body_names=".*"),
        "static_friction_range":(.5,1.2), "dynamic_friction_range":(.4,.9),
        "restitution_range":(0.,.05), "num_buckets":32, "make_consistent":True})
    mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot",body_names="base"),
        "mass_distribution_params":(-.5,1.5), "operation":"add"})
    payload = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot",body_names="payload"),
        "mass_distribution_params":(.05,.5), "operation":"abs", "recompute_inertia":True})
    payload_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot",body_names="payload"),
        "com_range":{"x":(-.03,.03),"y":(-.03,.03),"z":(-.02,.02)}})
    gains = EventTerm(func=mdp.randomize_actuator_gains, mode="reset", params={
        "asset_cfg":SceneEntityCfg("robot",joint_names=".*_(hip|thigh|calf)_joint"),
        "stiffness_distribution_params":(.9,1.1),"damping_distribution_params":(.9,1.1),"operation":"scale"})


@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.
    decimation = spec.DECIMATION
    action_space = 12
    observation_space = spec.FRAME_SIZE*spec.HISTORY
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=spec.DT, render_interval=spec.DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6,restitution=0.))
    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=4., replicate_physics=True)
    robot = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    events = EventCfg()
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane",
        collision_group=-1, physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6))
    # Ground truth is for rewards/guards/evaluation, never actor input.
    height_scanner = RayCasterCfg(prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,1.)), ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=.1,size=(.6,.6)),
        mesh_prim_paths=["/World/ground"], debug_vis=False)
    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3, update_period=spec.DT, track_air_time=True)
    command_curriculum_steps = 200000
    noise = True
    random_commands = True
    random_arm = True
    arm_motion = True
    termination_contact_force = 5.
    arm_tracking_limit = .12
    min_ground_clearance = .08


@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground",terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(curriculum=True,size=(8.,8.),border_width=12.,
            num_rows=8,num_cols=10,horizontal_scale=.1,vertical_scale=.005,slope_threshold=.75,
            use_cache=False, sub_terrains={
                "flat":terrain_gen.MeshPlaneTerrainCfg(proportion=.2),
                "rough":terrain_gen.HfRandomUniformTerrainCfg(proportion=.2,
                    noise_range=(.005,.04),noise_step=.005,border_width=.25),
                "slope":terrain_gen.HfPyramidSlopedTerrainCfg(proportion=.15,
                    slope_range=(0.,.25),platform_width=2.,border_width=.25),
                "slope_down":terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=.15,
                    slope_range=(0.,.25),platform_width=2.,border_width=.25),
                "stairs":terrain_gen.MeshPyramidStairsTerrainCfg(proportion=.15,
                    step_height_range=(.02,.10),step_width=.3,platform_width=3.,border_width=1.,holes=False),
                "stairs_down":terrain_gen.MeshInvertedPyramidStairsTerrainCfg(proportion=.15,
                    step_height_range=(.02,.10),step_width=.3,platform_width=3.,border_width=1.,holes=False),
            }),max_init_terrain_level=0,collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6))

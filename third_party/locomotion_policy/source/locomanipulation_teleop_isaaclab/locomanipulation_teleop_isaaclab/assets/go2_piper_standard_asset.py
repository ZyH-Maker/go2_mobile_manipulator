"""Shared MJCF converted to USD; standard PiPER, fixed-open gripper, flange load."""
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
import policy_contract as spec

USD = Path(__file__).resolve().parent/"generated/go2_piper_standard.usd"
GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(USD), activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
            max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=2)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,0.32),
        joint_pos=dict(zip(spec.LEGS+spec.ARMS, spec.HOME_LEGS+spec.HOME_ARM)),
        joint_vel={".*":0.}),
    actuators={
        "legs": DelayedPDActuatorCfg(joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            effort_limit={".*_hip_joint":23.7,".*_thigh_joint":23.7,".*_calf_joint":45.43},
            velocity_limit=30.1, stiffness=spec.LEG_KP, damping=spec.LEG_KD,
            min_delay=0, max_delay=1),
        "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"],
            effort_limit_sim=20., velocity_limit_sim=1., stiffness=spec.ARM_KP, damping=spec.ARM_KD),
    }, soft_joint_pos_limit_factor=0.95)

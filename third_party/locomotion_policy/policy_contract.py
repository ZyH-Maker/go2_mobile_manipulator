"""Standard-PiPER policy ABI. No Isaac/ROS imports; shared by training and sim2sim."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SCHEMA = "go2-standard-piper-blind-v1"
LEGS = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR"))
ARMS = tuple(f"joint{i}" for i in range(1, 7))
HOME_LEGS = (0.,)*4 + (0.9,)*4 + (-1.8,)*4
HOME_ARM = (0., 1.0, -1.2, 0., 0.2, 0.)
COMMAND_LOW = (-0.5, -0.25, -0.5, -0.25, -0.08)
COMMAND_HIGH = (0.5, 0.25, 0.5, 0.25, 0.)
COMMAND_RATE = (0.5, 0.5, 0.8, 0.3, 0.04)
DT = 0.005
DECIMATION = 4
CONTROL_DT = DT * DECIMATION
HISTORY = 5
FRAME_SIZE = 60
ACTION_SCALE = 0.5
ACTION_CLIP = 6.
ACTION_ALPHA = 0.8
LEG_KP, LEG_KD = 20., 1.5
ARM_KP, ARM_KD = 200., 10.
ARM_SPEED = 25. * 3.141592653589793 / 180.
ARM_ACCEL = 50. * 3.141592653589793 / 180.
BASE_HEIGHT = 0.30
MODEL = ROOT / "deploy/mujoco/models/go2_piper_standard.xml"


def manifest():
    return dict(schema=SCHEMA, robot="Go2+standard-PiPER", vision=False,
                input_size=FRAME_SIZE*HISTORY, frame_size=FRAME_SIZE, history=HISTORY,
                output_size=12, observation_order=["linear_velocity_b:3", "angular_velocity_b:3",
                "projected_gravity_b:3", "vx_vy_wz:3", "pitch_height_offset:2",
                "leg_position_error:12", "leg_velocity:12", "previous_raw_action:12",
                "gait_phase:4", "arm_position_error:6"],
                history_order="oldest_to_newest", leg_names=LEGS, arm_names=ARMS,
                default_leg_position=HOME_LEGS, default_arm_position=HOME_ARM,
                command_low=COMMAND_LOW, command_high=COMMAND_HIGH,
                command_rate=COMMAND_RATE, dt=DT, decimation=DECIMATION,
                action_scale=ACTION_SCALE, action_clip=ACTION_CLIP, action_alpha=ACTION_ALPHA,
                leg_pd=[LEG_KP, LEG_KD], arm_pd=[ARM_KP, ARM_KD],
                arm_speed=ARM_SPEED, arm_acceleration=ARM_ACCEL,
                base_height=BASE_HEIGHT, pitch_reference="world_horizontal",
                height_reference="mean_local_ground_plus_offset", gripper="fixed_open_0.04m",
                model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest())


def write_manifest(directory):
    path = Path(directory)/"policy_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=2)+"\n")
    return path


def validate_manifest(path):
    actual = json.loads(Path(path).read_text())
    expected = json.loads(json.dumps(manifest()))
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Policy contract mismatch: {key}; export with this source/model revision")
    return actual

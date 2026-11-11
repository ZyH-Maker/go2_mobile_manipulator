# Third-party components

This repository combines locally maintained Go2/PiPER integration code with the components below. The project MIT license covers original integration work; third-party code, models and SDK retain their own notices and license terms. Copyright notices in source files are retained. Original READMEs and Git histories were removed during consolidation; required license files remain.

| Component / source | Local location | License / notice |
|---|---|---|
| [LocoPilot](https://github.com/iit-DLSLab/LocoPilot) | Camera references and base deployment design | `third_party/licenses/LocoPilot.LICENSE` (MIT) |
| [Gamepad_PiPER](https://github.com/kehuanjack/Gamepad_PiPER) | Adapted arm recovery, keyboard/IK code; Piper model assets | `third_party/licenses/Gamepad_PiPER.LICENSE` (MIT) |
| [locomanipulation-teleop-isaaclab](https://github.com/Giovadess/locomanipulation-teleop-isaaclab) | `third_party/locomotion_policy`, `models/policies` | `third_party/locomotion_policy/LICENSE` (BSD 3-Clause) |
| [MUSE](https://github.com/Giovadess/muse) | `third_party/state_estimator` | `third_party/state_estimator/LICENSE` (BSD 3-Clause) |
| [unitree-ros2-dls](https://github.com/Giovadess/unitree-ros2-dls) | `third_party/go2_driver` | Package license declarations and retained message licenses |
| [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) | `third_party/go2_driver/unitree_ros2` | Its retained `LICENSE` (BSD 3-Clause) |
| [AgileX arm URDF](https://github.com/agilexrobotics/agx_arm_urdf) | `models/piper_limits.urdf` | `third_party/licenses/AgileX_URDF.LICENSE` (MIT) |
| [PiPER SDK](https://github.com/agilexrobotics/piper_sdk), 0.6.2 | `third_party/piper_sdk_runtime` | `piper_sdk-0.6.2.dist-info/licenses/LICENSE` |

Locked upstream archive revisions are recorded in `third_party/source_versions.json` and `third_party/completed_dependencies.json`. Local changes include Foxy compatibility, managed HAL handover, independent arm executor, ROS target interface, Chinese terminal UI and the consolidated layout. The original locomotion policy remains trained for Piper L; this repository uses the standard Piper feedback limits. The camera launch files are retained as reference and have not been validated in the current Foxy runtime.

The training subtree has additionally been adapted for standard PiPER: shared MJCF/USD generation from the vendored standard URDF/meshes, explicit finger-inertia corrections, collision-filtered arm motion graphs, five-axis body commands, Flat/Rough-only PPO tasks, evaluation/export contracts, and MuJoCo sim2sim. Obsolete upstream training/vision/RMA helpers and duplicate deployment demos have been removed. The original Piper L inference dependencies remain solely for the existing top-level hardware entry point. No newly trained policy is included. See `third_party/locomotion_policy/README.md` for model assumptions and validation status.

"""Offline dependency checks: no ROS nodes, CAN connections, or commands."""
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess

root = Path(__file__).resolve().parents[1]
result = {"python": sys.version, "checks": {}, "hardware_commands_sent": 0}


def check(name, fn):
    try:
        result["checks"][name] = {"ok": True, "detail": fn()}
    except Exception as exc:
        result["checks"][name] = {"ok": False, "error": repr(exc)}


# Match policy_controller's import order on aarch64: reserve OpenMP TLS
# before ROS/MuJoCo load their libraries.
check("torch", lambda: str(importlib.import_module("torch").__file__))
for module in [
    "rclpy", "unitree_go.msg", "unitree_api.msg", "dls2_interface.msg",
    "mujoco",
    "onnxruntime",
]:
    check(module, lambda module=module: str(importlib.import_module(module).__file__))


def model_check():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(
        root / "third_party/locomotion_policy/deploy/mujoco/models/scene_flat.xml"
    ))
    return {"nq": model.nq, "nv": model.nv, "nu": model.nu,
            "mujoco_version": mujoco.__version__}


def inference_check():
    import numpy as np
    import onnxruntime as ort
    policy = root / "models/policies/arm_moving_clock/exported/policy.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    output = session.run(None, {"obs": np.zeros((1, 300), dtype=np.float32)})[0]
    assert output.shape == (1, 12) and np.isfinite(output).all()
    return {"input": session.get_inputs()[0].shape, "output": list(output.shape)}


def wrapper_check():
    import numpy as np
    import mujoco
    deploy = root / "third_party/locomotion_policy/deploy"
    sys.path.insert(0, str(deploy))
    from locomotion_policy_wrapper import LocomotionPolicyWrapper
    model = mujoco.MjModel.from_xml_path(str(deploy / "mujoco/models/scene_flat.xml"))
    wrapper = LocomotionPolicyWrapper(model)
    for _ in range(5):
        output = wrapper.compute_control(
            base_pos=np.array([0., 0., 0.3]), base_ori_euler_xyz=np.zeros(3),
            base_quat_wxyz=np.array([1., 0., 0., 0.]),
            base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3),
            heading_orientation_SO3=np.eye(3),
            joints_pos_leg=wrapper.default_joint_pos_leg.copy(),
            joints_vel_leg=np.zeros(12), ref_base_lin_vel=np.zeros(3),
            ref_base_ang_vel=np.zeros(3), ref_pose_command=np.zeros(2),
            joints_pos_arm=wrapper.default_joint_pos_arm.copy())
        assert output.shape == (12,) and np.isfinite(output).all()
    return {"policy_hz": wrapper.RL_FREQ, "history": list(wrapper._observation_history.shape),
            "target_shape": list(output.shape)}


check("mujoco_model", model_check)
check("onnx_inference", inference_check)
check("upstream_policy_wrapper", wrapper_check)


def binary_check(relative_path):
    path = root / relative_path
    assert path.is_file() and os.access(str(path), os.X_OK), str(path)
    linked = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=True)
    missing = [line for line in linked.stdout.splitlines() if "not found" in line]
    assert not missing, missing
    return {"path": str(path), "missing_libraries": missing, "executed": False}


check("managed_go2_hal_binary", lambda: binary_check("runtime/go2_install/go2_hal/bin/managed_go2_hal"))
check("muse_binary", lambda: binary_check("runtime/estimator_install/state_estimator/lib/state_estimator/state_estimator_node"))


def octomap_alias_check():
    directory = root/'runtime/venv/lib/python3.8/site-packages/cmeel.prefix/lib'
    details = {}
    for name in ['liboctomap', 'liboctomath']:
        paths = sorted(directory.glob(name+'.so*'))
        assert paths, 'Missing '+name
        identities = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
        assert len(identities)==1, 'Duplicate ELF copies can cause double-free: '+str(paths)
        details[name] = str(paths[0].resolve())
    return details


check("octomap_library_aliases", octomap_alias_check)
result["all_passed"] = all(v["ok"] for v in result["checks"].values())
result["limitation"] = "Dependency/import/inference checks only; no closed-loop or hardware validation."
output_path = root / "runtime/reports/runtime_check.json"
output_path.parent.mkdir(parents=True,exist_ok=True)
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
sys.exit(0 if result["all_passed"] else 1)

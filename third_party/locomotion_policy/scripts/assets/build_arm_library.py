"""Generate a connected arm motion graph checked against MJCF contacts.

Checks whole sampled edges at nominal stance and several body pitch/height poses.
This is a sampling filter, not a runtime collision guarantee on moving rough terrain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import policy_contract as spec


def generate(count=32, seed=42, samples=25):
    model = mujoco.MjModel.from_xml_path(str(spec.MODEL.parent/"scene_standard_flat.xml"))
    data = mujoco.MjData(model)
    leg_q = [model.joint(n).qposadr[0] for n in spec.LEGS]
    arm_q = [model.joint(n).qposadr[0] for n in spec.ARMS]
    limits = np.array([model.joint(n).range for n in spec.ARMS])
    rng = np.random.default_rng(seed)
    arm_bodies = {model.body(n).id for n in ("piper_base", *(f"link{i}" for i in range(1,9)), "gripper_base", "payload")}

    def clear(q):
        for pitch, height in ((0., .30), (-.25,.30), (.25,.30), (-.25,.22), (.25,.22), (0.,.22)):
            data.qpos[:] = 0
            data.qpos[2] = height
            data.qpos[3:7] = (math.cos(pitch/2), 0, math.sin(pitch/2), 0)
            data.qpos[leg_q] = spec.HOME_LEGS
            data.qpos[arm_q] = q
            mujoco.mj_forward(model, data)
            for c in data.contact:
                a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if c.dist < .003 and (a in arm_bodies or b in arm_bodies):
                    return False
        return True

    poses = [np.array(spec.HOME_ARM)]
    if not clear(poses[0]):
        raise ValueError("Home pose collides: correct mount/home/model before generating motions")
    edges = [[]]
    attempts = 0
    while len(poses) < count and attempts < count*500:
        attempts += 1
        target = rng.uniform(limits[:,0]+.06, limits[:,1]-.06)
        if not clear(target):
            continue
        # Expand the existing connected graph, avoiding isolated valid targets.
        candidates = np.argsort([np.linalg.norm(q-target) for q in poses])[:4]
        source = None
        for i in candidates:
            if all(clear(poses[i]*(1-t)+target*t) for t in np.linspace(0,1,samples)):
                source = int(i)
                break
        if source is None:
            continue
        j = len(poses)
        poses.append(target); edges.append([source]); edges[source].append(j)
    if len(poses) < count:
        raise RuntimeError(f"Only {len(poses)}/{count} valid connected poses; inspect collision geometry")
    output = ROOT/"config/arm_motion_library.json"
    output.write_text(json.dumps(dict(schema=spec.SCHEMA,
        model_sha256=hashlib.sha256(spec.MODEL.read_bytes()).hexdigest(), seed=seed,
        path_samples=samples, clearance_m=.003, poses=[q.tolist() for q in poses], edges=edges,
        validation="nominal legs; pitch +/-0.25 rad; base height 0.22/0.30 m; flat ground"), indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    if args.poses < 2 or args.samples < 3:
        parser.error("need >=2 poses and >=3 samples")
    generate(args.poses, args.seed, args.samples)

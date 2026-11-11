"""Build a shared MJCF from vendored Go2 and standard PiPER URDF (stdlib only)."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import policy_contract as spec


def numbers(value):
    return [float(x) for x in value.split()]


def quaternion(rpy):
    r, p, y = [x/2 for x in numbers(rpy)]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ' '.join(map(str, (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)))


def build(mount=(0., 0., 0.07)):
    directory = spec.MODEL.parent
    project = ROOT.parents[1]
    urdf_path = project/"models/piper/piper.urdf"
    urdf = ET.parse(urdf_path).getroot()
    # Start with the Go2 used by the existing policy: preserve its inertias and geometry.
    root = ET.parse(directory/"go2_piper_l.xml").getroot()
    root.set("model", "go2_standard_piper")
    root.find("option").set("timestep", str(spec.DT))
    root.find("option").set("integrator", "implicitfast")
    base = root.find("worldbody/body[@name='base']")
    for body in list(base.findall("body")):
        if body.get("name") in ("base_link", "link1"):
            base.remove(body)
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if "piper_l/" in mesh.get("file", ""):
            asset.remove(mesh)
    mesh_names = {mesh.get('name', Path(mesh.get('file')).stem) for mesh in asset.findall('mesh')}
    for geom in list(base.findall('geom')):
        if geom.get('mesh') and geom.get('mesh') not in mesh_names:
            base.remove(geom)
    for tag in ("actuator", "sensor", "keyframe", "equality", "contact"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    actuator = ET.SubElement(root, "actuator")
    for name in spec.LEGS:
        limit = 45.43 if "calf" in name else 23.7
        ET.SubElement(actuator, "motor", name=name, joint=name, gear="1", ctrlrange=f"{-limit} {limit}")
    contact = ET.SubElement(root, "contact")
    corrections = []
    links = {x.get("name"): x for x in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)

    def add_link(parent, name, joint=None):
        link = links[name]
        body_name = "piper_base" if name == "base_link" else name
        attrs = dict(name=body_name)
        if joint is None:
            attrs["pos"] = ' '.join(map(str, mount))
        else:
            origin = joint.find("origin")
            attrs.update(pos=origin.get("xyz", "0 0 0"), quat=quaternion(origin.get("rpy", "0 0 0")))
        body = ET.SubElement(parent, "body", **attrs)
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        origin, inertia = inertial.find("origin"), inertial.find("inertia")
        attrs = dict(mass=str(mass), pos=origin.get("xyz", "0 0 0"))
        if name in ("link7", "link8"):
            # The supplied URDF has negative Iyy. Use a documented 20x60x10mm box.
            dims = (0.02, 0.06, 0.01)
            diag = [mass*(dims[(i+1)%3]**2+dims[(i+2)%3]**2)/12 for i in range(3)]
            attrs["diaginertia"] = ' '.join(map(str, diag))
            corrections.append(dict(link=name, source_iyy=inertia.get("iyy"), box_m=dims, diaginertia=diag))
        else:
            if any(float(inertia.get(k)) <= 0 for k in ("ixx", "iyy", "izz")):
                raise ValueError(f"Invalid inertia in {name}")
            attrs["fullinertia"] = ' '.join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        ET.SubElement(body, "inertial", **attrs)
        mesh_name = "standard_"+name
        mesh_path = project/"models/piper/meshes"/Path(link.find("visual/geometry/mesh").get("filename")).name
        ET.SubElement(asset, "mesh", name=mesh_name, file=os.path.relpath(mesh_path, directory))
        ET.SubElement(body, "geom", name=mesh_name+"_visual", type="mesh", mesh=mesh_name,
                      contype="0", conaffinity="0", group="2", rgba="0.7 0.75 0.8 1")
        # Separate contact geom: both engines use a convex hull per link.
        ET.SubElement(body, "geom", name=mesh_name+"_collision", type="mesh", mesh=mesh_name,
                      contype="1", conaffinity="1", group="3", friction="0.6 0.005 0.0001")
        if joint is not None and joint.get("type") == "revolute":
            joint_name = joint.get("name")
            limit = joint.find("limit")
            # Match the real executor's stricter J6 range (120 degrees).
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            if joint_name == "joint6":
                lo, hi = -math.radians(120), math.radians(120)
            ET.SubElement(body, "joint", name=joint_name, type="hinge", axis=joint.find("axis").get("xyz"),
                          range=f"{lo} {hi}", damping="0.1", armature="0.01", frictionloss="0.1")
            # Conservative explicit arm effort setting, not the URDF's unvalidated 100 Nm.
            ET.SubElement(actuator, "position", name=joint_name, joint=joint_name,
                          kp=str(spec.ARM_KP), kv=str(spec.ARM_KD),
                          ctrlrange=f"{lo} {hi}", forcerange="-20 20")
        for child_joint in children.get(name, []):
            child = add_link(body, child_joint.find("child").get("link"), child_joint)
            if child_joint.get("type") == "prismatic":
                # Fixed-open fingers: this policy controls legs only, no grasping task.
                axis = numbers(child_joint.find("axis").get("xyz"))
                q = 0.02 if child_joint.get("name") == "joint7" else -0.02
                # Translation is in the child joint frame; rotate it to the parent.
                w, x, y, z = numbers(child.get("quat"))
                rot = ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                       (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                       (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
                pos = numbers(child.get("pos"))
                child.set("pos", ' '.join(str(pos[i]+q*sum(rot[i][j]*axis[j] for j in range(3))) for i in range(3)))
            ET.SubElement(contact, "exclude", body1=body_name, body2=child.get("name"))
        return body

    piper = add_link(base, "base_link")
    ET.SubElement(contact, "exclude", body1="base", body2="piper_base")
    # Payload is a separate inertial body so its mass/COM can be randomized at the flange.
    flange = piper.find(".//body[@name='link6']")
    payload = ET.SubElement(flange, "body", name="payload", pos="0 0 0.10")
    ET.SubElement(payload, "inertial", mass="0.1", pos="0 0 0", diaginertia="0.0001 0.0001 0.0001")
    ET.SubElement(flange, "site", name="tcp", pos="0 0 0.1358", size="0.008")
    base.set("pos", "0 0 0.32")
    ET.indent(root)
    ET.ElementTree(root).write(spec.MODEL, encoding="unicode")
    report = dict(source_urdf="models/piper/piper.urdf", mount_xyz=mount,
                  gripper="fixed_open_0.04m", inertia_corrections=corrections,
                  arm_effort_limit_nm=20, hardware_parameters_validated=False)
    spec.MODEL.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    for terrain in ("flat", "rough"):
        scene = ET.Element("mujoco", model="standard_piper_"+terrain)
        ET.SubElement(scene, "include", file=spec.MODEL.name)
        world = ET.SubElement(scene, "worldbody")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="30 30 0.1", rgba="0.4 0.45 0.4 1", friction="0.8 0.005 0.0001")
        if terrain == "rough":
            # Spawn on flat ground; approach a reproducible 5/7/9cm staircase in +x.
            for i, height in enumerate((0.05, 0.12, 0.21, 0.12, 0.05)):
                ET.SubElement(world, "geom", name=f"step_{i}", type="box", size=f"0.18 1 {height/2}",
                              pos=f"{1.5+i*0.36} 0 {height/2}", rgba="0.5 0.5 0.5 1")
        ET.indent(scene)
        ET.ElementTree(scene).write(directory/f"scene_standard_{terrain}.xml", encoding="unicode")
    print(spec.MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount", nargs=3, type=float, default=(0., 0., 0.07), metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    build(tuple(args.mount))

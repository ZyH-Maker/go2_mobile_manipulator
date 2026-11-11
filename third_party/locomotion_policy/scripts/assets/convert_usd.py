"""Convert the shared standard-PiPER MJCF once in the Isaac Lab environment."""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
try:
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import policy_contract as spec
    import hashlib
    target = root/"source/locomanipulation_teleop_isaaclab/locomanipulation_teleop_isaaclab/assets/generated"
    converter = MjcfConverter(MjcfConverterCfg(asset_path=str(spec.MODEL),
        usd_dir=str(target), usd_file_name="go2_piper_standard.usd", fix_base=False,
        self_collision=True, import_inertia_tensor=True, import_sites=True,
        make_instanceable=True, force_usd_conversion=True))
    (target/"model.sha256").write_text(hashlib.sha256(spec.MODEL.read_bytes()).hexdigest()+"\n")
    print(converter.usd_path)
finally:
    app.app.close()

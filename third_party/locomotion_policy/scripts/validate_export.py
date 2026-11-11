"""Check a policy bundle, JIT/ONNX numerical parity and the runtime adapter."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'deploy')]
import policy_contract as spec
from standard_runtime import StandardPolicy


def validate(directory, samples=64):
    directory = Path(directory)
    spec.validate_manifest(directory/'policy_contract.json')
    runtime = StandardPolicy(directory)
    jit = torch.jit.load(str(directory/'policy.pt'),map_location='cpu').eval()
    rng = np.random.default_rng(42)
    largest = 0.
    with torch.inference_mode():
        for i in range(samples):
            obs = np.zeros((1,300),np.float32) if i==0 else rng.normal(0,.3,(1,300)).astype(np.float32)
            a = jit(torch.from_numpy(obs)).numpy()
            b = runtime.session.run(None,{runtime.input_name:obs})[0]
            if a.shape!=(1,12) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid policy output')
            largest = max(largest,float(np.abs(a-b).max()))
            if not np.allclose(a,b,atol=1.e-5,rtol=1.e-4):
                raise ValueError(f'JIT/ONNX mismatch, max abs={largest}')
            runtime.infer(obs[0,-60:])
    return dict(schema=spec.SCHEMA,samples=samples,max_abs_jit_onnx=largest,
                passed=True,note='Numerical export validation only, not locomotion capability')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    args=p.parse_args()
    result=validate(args.policy)
    (args.policy/'export_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

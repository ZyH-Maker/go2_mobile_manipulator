"""NumPy/ONNX adapter for the new policy; does not import ROS or legacy config."""
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_contract as spec


class StandardPolicy:
    def __init__(self, directory):
        directory = Path(directory)
        spec.validate_manifest(directory/'policy_contract.json')
        self.session = ort.InferenceSession(str(directory/'policy.onnx'), providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs)!=1 or inputs[0].shape != [1,300] or inputs[0].type!='tensor(float)':
            raise ValueError('Expected one float32 (1,300) actor input')
        if len(outputs)!=1 or outputs[0].shape != [1,12]:
            raise ValueError('Expected one (1,12) action output')
        self.input_name = inputs[0].name
        self.reset()

    def reset(self):
        self.history = np.zeros((spec.HISTORY,spec.FRAME_SIZE), dtype=np.float32)
        self.previous = np.zeros(12)

    def infer(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape!=(60,) or not np.isfinite(frame).all():
            raise ValueError('Expected a finite 60-element observation')
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        action = self.session.run(None,{self.input_name:self.history.reshape(1,-1)})[0][0]
        if not np.isfinite(action).all():
            raise ValueError('Non-finite ONNX action')
        action = action.clip(-spec.ACTION_CLIP,spec.ACTION_CLIP)
        target = np.asarray(spec.HOME_LEGS)+spec.ACTION_SCALE*(spec.ACTION_ALPHA*action+(1-spec.ACTION_ALPHA)*self.previous)
        self.previous = action.copy()
        return target


def observation(linear_b, angular_b, gravity_b, commands, legs, leg_velocity, previous, phase, arm):
    return np.concatenate((linear_b,angular_b,gravity_b,commands,
        np.asarray(legs)-spec.HOME_LEGS,leg_velocity,previous,phase,
        np.asarray(arm)-spec.HOME_ARM)).astype(np.float32)

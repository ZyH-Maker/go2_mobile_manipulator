"""Exercise MUSE startup, feedback and shutdown. Live mode only subscribes to hardware."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--interface', default='eth1')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--prefix', type=Path)
    args = parser.parse_args()
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
    os.environ['ROS_DOMAIN_ID'] = os.environ.get('ROS_DOMAIN_ID', '0') if args.live else '228'
    os.environ['ROS_LOCALHOST_ONLY'] = '0' if args.live else '1'
    if args.live:
        os.environ['CYCLONEDDS_URI'] = ('<CycloneDDS><Domain><General><Interfaces>'
            '<NetworkInterface name="%s"/></Interfaces></General></Domain></CycloneDDS>' % args.interface)
    else:
        os.environ['CYCLONEDDS_URI'] = ('<CycloneDDS><Domain><General>'
            '<NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    from dls2_interface.msg import BaseState
    from unitree_go.msg import LowState

    rclpy.init()
    node = Node('locopilot_muse_runtime_probe')
    samples, raw_samples = deque(maxlen=20000), deque(maxlen=20000)
    node.create_subscription(BaseState, '/base_state',
                             lambda msg: samples.append((time.monotonic(), msg)), 10)
    node.create_subscription(LowState, '/lowstate',
                             lambda msg: raw_samples.append((time.monotonic(), msg)), qos_profile_sensor_data)
    def pump(seconds):
        end = time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(node, timeout_sec=.002)
    pump(1.5)
    assert node.count_publishers('/base_state')==0, 'Another base-state estimator is already running'
    existing_lowcmd = node.count_publishers('/lowcmd')
    if args.live:
        assert raw_samples, 'No real /lowstate feedback on the selected interface/domain'
    else:
        assert existing_lowcmd==0, 'Unexpected leg controller in isolated test domain'
        assert node.count_publishers('/lowstate')==0, 'Unexpected hardware publisher in isolated test domain'
        raw = LowState()
        raw.imu_state.quaternion = [1., 0., 0., 0.]
        raw.imu_state.accelerometer = [0., 0., 9.81]
        raw.foot_force = [50]*4
        for i in range(12):
            raw.motor_state[i].q = float([0., .9, -1.8][i%3])
        publisher = node.create_publisher(LowState, '/lowstate', 10)
        node.create_timer(.002, lambda: publisher.publish(raw))
    prefix = args.prefix or Path(get_package_prefix('state_estimator'))
    share = Path(get_package_share_directory('state_estimator'))
    binary = prefix/'lib/state_estimator/state_estimator_node'
    output = ROOT/'runtime/reports'/('muse_live' if args.live else 'muse_synthetic')
    output.mkdir(parents=True, exist_ok=True)
    stream = (output/'runtime.log').open('w')
    process = subprocess.Popen([str(binary), '--ros-args', '-p', 'config_dir:='+str(share/'config'),
                                '-p', 'launch_dir:='+str(share/'launch')],
                               stdout=stream, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+10
        while not samples and process.poll() is None and time.monotonic()<deadline:
            pump(.05)
        assert samples, 'MUSE produced no /base_state; inspect '+str(output/'runtime.log')
        samples.clear();raw_samples.clear()
        pump(args.seconds)
        assert process.poll() is None, 'MUSE exited while receiving feedback'
        assert samples and raw_samples, 'Feedback stream missing'
        times = [s[0] for s in samples]
        vectors = [list(m.pose.position)+list(m.pose.orientation)+list(m.velocity.linear)+list(m.velocity.angular)
                   for _,m in samples]
        finite = all(math.isfinite(float(x)) for v in vectors for x in v)
        norms = [math.sqrt(sum(float(x)**2 for x in m.pose.orientation)) for _,m in samples]
        linear_speeds = [math.sqrt(sum(float(x)**2 for x in m.velocity.linear)) for _,m in samples]
        rate = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0
        gaps = [b-a for a,b in zip(times,times[1:])]
        last = samples[-1][1]
        x,y,z,w = map(float,last.pose.orientation)
        estimated_rpy = [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                         math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
                         math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
        raw_rpy = list(raw_samples[-1][1].imu_state.rpy)
        rp_error = [math.degrees((a-b+math.pi)%(2*math.pi)-math.pi)
                    for a,b in zip(estimated_rpy[:2],raw_rpy[:2])]
        metrics = {'scope': 'live feedback only' if args.live else 'loopback synthetic LowState',
            'sample_seconds':args.seconds, 'base_messages':len(samples), 'base_hz':rate,
            'raw_messages':len(raw_samples), 'all_values_finite':finite,
            'quaternion_norm_range':[min(norms),max(norms)],
            'maximum_receipt_gap_s':max(gaps) if gaps else None,
            'maximum_linear_speed_m_s':max(linear_speeds),
            'last_position':list(last.pose.position), 'last_orientation_xyzw':list(last.pose.orientation),
            'last_velocity':list(last.velocity.linear),
            'last_raw_rpy':raw_rpy, 'estimated_rpy_rad':estimated_rpy,
            'roll_pitch_difference_deg':rp_error,
            'existing_lowcmd_publishers_before':existing_lowcmd,
            'lowcmd_publishers':node.count_publishers('/lowcmd'),
            'probe_publishes_lowcmd':False,
            'hardware_motion_commands_sent':0}
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        metrics['exit_code'] = process.returncode
        metrics['passed'] = (finite and .99<min(norms)<=max(norms)<1.01 and rate>=45
                             and gaps and max(gaps)<.1 and max(abs(v) for v in rp_error)<5
                             and process.returncode==0)
        text = json.dumps(metrics, indent=2, default=float)
        (output/'result.json').write_text(text)
        print(text)
        assert metrics['passed'], 'MUSE runtime check failed; inspect result/log'
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        stream.close()
        node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    main()

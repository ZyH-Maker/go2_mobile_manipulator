#!/usr/bin/env bash
# All tests run offline or over isolated loopback ROS domains.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
cd "$GMM_ROOT"
python -B -m unittest discover -s arm/tests -p 'test_*.py'
python -B -m unittest discover -s locomotion/tests -p test_base_core.py
python -B arm/tests/test_arm_interface_ros.py
python -B locomotion/tests/test_base_ros.py --arm-executor --stand-offset --recent-lowcmd

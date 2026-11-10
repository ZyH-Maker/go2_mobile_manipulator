#!/usr/bin/env bash
# Load the shared project environment. Starts no nodes and sends no commands.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请 source 本文件" >&2
    exit 1
fi
GMM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/ros/foxy/setup.bash ]]; then
    echo "请先进入 ROS Foxy 容器：docker exec -it go2_piper bash" >&2
    return 1
fi
source /opt/ros/foxy/setup.bash || return 1
# Existing container's DDS/message overlay is optional after tools/build.sh.
if [[ -f "${UNITREE_ROS_SETUP:-/workspace/unitree_ros2/cyclonedds_ws/install/setup.bash}" ]]; then
    source "${UNITREE_ROS_SETUP:-/workspace/unitree_ros2/cyclonedds_ws/install/setup.bash}" || return 1
fi
if [[ -f "$GMM_ROOT/runtime/venv/bin/activate" ]]; then
    source "$GMM_ROOT/runtime/venv/bin/activate" || return 1
fi
if [[ "${GMM_BUILD_MODE:-0}" != 1 ]]; then
    for gmm_overlay in unitree_install go2_install estimator_install; do
        if [[ -f "$GMM_ROOT/runtime/$gmm_overlay/local_setup.bash" ]]; then
            source "$GMM_ROOT/runtime/$gmm_overlay/local_setup.bash" || return 1
        fi
    done
fi
gmm_cmeel="$GMM_ROOT/runtime/venv/lib/python3.8/site-packages/cmeel.prefix"
if [[ -d "$gmm_cmeel" ]]; then
    export CMAKE_PREFIX_PATH="$gmm_cmeel${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="$gmm_cmeel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth1"/></Interfaces></General></Domain></CycloneDDS>'
export GMM_ROOT
unset gmm_overlay gmm_cmeel

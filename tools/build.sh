#!/usr/bin/env bash
# Build only: never starts hardware-facing nodes.
set -eo pipefail
GMM_BUILD_MODE=1 source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
python "$GMM_ROOT/tools/repair_octomap.py"
cd "$GMM_ROOT"
gmm_build() {
    local gmm_name="$1"
    shift
    python -m colcon --log-base "runtime/build_logs/$gmm_name" build \
        --build-base "runtime/build/$gmm_name" --install-base "runtime/${gmm_name}_install" \
        --executor sequential "$@" --cmake-args -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
    source "$GMM_ROOT/runtime/${gmm_name}_install/local_setup.bash"
}
gmm_build unitree --base-paths third_party/go2_driver/unitree_ros2/cyclonedds_ws/src/unitree \
    --packages-select unitree_go unitree_api
gmm_build go2 --base-paths third_party/go2_driver/ros2_ws/src \
    --packages-select dls2_interface go2_hal
gmm_build estimator --base-paths third_party/state_estimator/muse_ws/src \
    --packages-ignore dls2_interface unitree_go muse_point_lio

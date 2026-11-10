#!/usr/bin/env bash
# Install Python dependencies only. Requires the ROS Foxy/PyTorch container.
set -eo pipefail
GMM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$GMM_ROOT/runtime/reports" "$GMM_ROOT/runtime/logs/arm" "$GMM_ROOT/runtime/logs/base"
python3 -m venv --without-pip --system-site-packages "$GMM_ROOT/runtime/venv"
gmm_python="$GMM_ROOT/runtime/venv/bin/python"
if ! "$gmm_python" -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --target "$GMM_ROOT/runtime/venv/lib/python3.8/site-packages" 'pip<25.1'
fi
"$gmm_python" -m pip install -r "$GMM_ROOT/requirements.txt"
"$gmm_python" "$GMM_ROOT/tools/repair_octomap.py"
echo "Python 环境已准备；接下来运行 bash tools/build.sh。"

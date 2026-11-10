# 开发说明

仓库根目录：`/home/unitree/go2_piper/go2_mobile_manipulator`。所有运行路径以根 README 为准，旧 LocoPilot/Gamepad_PiPER 目录已经移除。

## 本地验证

```bash
source /workspace/go2_mobile_manipulator/env.sh
cd /workspace/go2_mobile_manipulator
python tools/check_runtime.py
bash tools/test.sh
```

`arm/tests/` 检查反馈解析、接管/恢复、轨迹、中文界面和模拟 CAN/ROS 流程；`locomotion/tests/` 检查 PD/遥控/状态机和真实 ONNX 的回环流程。两个回环测试固定不同 domain，仅使用 lo。

## 代码分工

- `arm/arm_config.py`：机械臂恢复、限位、键盘参数。
- `arm/core/state_machine.py`：CAN 状态解析和位置接管/恢复。
- `arm/arm_backend.py`：固定 SDK 与执行状态机适配。
- `arm/run_executor.py`：轨迹插值调度、服务和真实反馈。
- `arm/keyboard_node.py`：两个键盘共用的 ROS 目标节点。
- `locomotion/base_controller.py`：Go2 准备、站立与策略协调。
- `locomotion/base_core.py`：遥控、反馈契约、状态机与 PD。
- `third_party/`：保留的第三方实现，修改后需同步记录来源与变化。

许可证和来源信息属于发布内容。顶层 LICENSE 适用于本项目原创整合代码，第三方代码和模型保留各自许可证。

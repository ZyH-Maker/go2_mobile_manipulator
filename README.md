# Go2 Mobile Manipulator

Go2 四足底盘 + PiPER 机械臂的 ROS 2 控制项目。底盘运行已有强化学习策略，机械臂通过独立执行节点接收关节轨迹；键盘、后续 VLA 或强化学习控制器可共用这个位置接口。

当前运行环境：Jetson / Ubuntu、ROS 2 Foxy、Python 3.8、CycloneDDS；使用现有 `go2_piper` 容器。机械臂 SDK 固定为 0.6.2，底盘策略为 `arm_moving_clock`。

## 目录

```text
go2_mobile_manipulator/
├── arm/                  # 机械臂执行节点、两种键盘、配置与测试
├── locomotion/           # Go2 接管、站立、策略与测试
├── camera/reference/    # 后续相机接入参考，当前不启动
├── models/               # 标准 Piper 限位、URDF/STL、已有底盘策略
├── third_party/          # 推理封装、HAL/MUSE 源码、SDK 及许可证
├── tools/                # 环境准备、编译、依赖检查、离线验证
├── docs/                 # 通用机械臂接口、开发和训练说明
├── env.sh                # 所有终端共用的环境
└── runtime/              # 本机虚拟环境、编译产物、日志；不包含在源码包中
```

## 运行

宿主机目录：`/home/unitree/go2_piper/go2_mobile_manipulator`。容器目录：`/workspace/go2_mobile_manipulator`。每个终端先运行 `docker exec -it go2_piper bash` 进入容器。

CAN 已 UP、1 Mbps 时跳过配置。否则在**宿主机**执行：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

终端 1，机械臂执行节点：

```bash
source /workspace/go2_mobile_manipulator/env.sh
python /workspace/go2_mobile_manipulator/arm/run_executor.py --can can0
```

终端 2，末端键盘：

```bash
source /workspace/go2_mobile_manipulator/env.sh
python /workspace/go2_mobile_manipulator/arm/keyboard_cartesian.py
```

也可把第二个程序替换为单关节键盘，二选一：

```bash
python /workspace/go2_mobile_manipulator/arm/keyboard_joint.py
```

只控制机械臂时运行以上两个程序即可。执行节点启动后只读，Enter 接管当前位置，R 明确申请复位/使能和轻微越界恢复。等中文界面显示“位置保持”后再运动。R 会实际运动，最多每轴恢复 5°；超过范围会拒绝。真正控制故障锁定在执行节点中，重启键盘不能解除，需要重启执行节点。

| 操作 | 末端键盘 | 单关节键盘 |
|---|---|---|
| 接管 / 继续 | Enter | Enter / E |
| 越界恢复 | R | R |
| 转底座 | A / D | 选中 J1 后 J / L |
| 前后 / 上下 | Q / E，W / S | — |
| J5 / J6 | J / L，U / O | 选关节后 J / L |
| 选择关节 / 步长 | — | 1～6；[ 0.2° / ] 0.5° |
| 六轴 Home | B | — |
| 取消并保持 | P | P |
| 快速停止并锁定 | 空格 | 空格 |
| 请求保持并退出 | X / Ctrl+C | X / Ctrl+C |

终端 3，可选的 Go2 强化学习策略：

```bash
source /workspace/go2_mobile_manipulator/env.sh
python /workspace/go2_mobile_manipulator/locomotion/run_base.py --interface eth1
```

先完成机械臂接管/恢复，再按遥控 L1 准备 Go2；五秒插值站立并到 READY 后，Y 启动策略。B 速度命令归零、策略继续，摇杆回中后 Y 恢复；L2/R2 停止底盘并锁定。准备 PD 60/5；默认策略 PD 20/1.5，可用 `--policy-kp` / `--policy-kd` 指定。

机械臂与底盘的停止独立。底盘程序只读真实机械臂角度，不切换机械臂模式、不发送机械臂目标。底盘接管后拒绝机械臂重新复位/使能，普通轨迹继续可用。底盘停止不自动恢复 Sport。

## 控制器和相机扩展

底盘策略的上游关系、论文来源、重新训练、台阶训练、更换机器人模型和部署适配，见 [底盘策略训练与部署说明](docs/policy_training.md)。

机械臂目标输入为 `/follower/arm_controller/joint_trajectory`，类型 `trajectory_msgs/msg/JointTrajectory`。反馈为 `/follower/feedback/joint_states`。只运行一个目标发布者，未来模型控制器替换键盘即可；详见 [机械臂接口](docs/arm_interface.md)。

`camera/reference/` 保留双 RealSense 相机与机械臂轨迹控制器的参考 launch。它们来自 Jazzy 方案，当前作为参考，不是 Foxy 下已验证的相机入口。后续相机节点可与现有机械臂执行节点并行，给模型控制器提供图像。

## 从源码准备环境

本机已经完成环境迁移和编译，不用重装。新机器需要 Ubuntu/ROS Foxy、ARM64 兼容 PyTorch、CycloneDDS、Eigen3、yaml-cpp 和 ROS 编译工具。现有容器提供这些系统依赖；源码包不包含 Docker 镜像或虚拟环境。

```bash
cd /workspace/go2_mobile_manipulator
bash tools/bootstrap.sh
bash tools/build.sh
source env.sh
python tools/check_runtime.py
bash tools/test.sh
```

`bootstrap.sh` 创建共享 Python 环境，`build.sh` 编译 Unitree 消息、Go2 HAL 与 MUSE。测试使用 lo 和独立 ROS domain、模拟 CAN，不驱动真机。本次迁移已通过 80 项单元测试、机械臂模拟 CAN/ROS、真实 ONNX 的底盘联合回环、键盘终端显示/退出和 MUSE 启动/反馈/退出验证；未发送真机控制指令。以上步骤在当前现有容器中验证；全新系统镜像尚未验收。

## 日志与模型

- `runtime/logs/arm/`：机械臂事件/轨迹日志。
- `runtime/logs/base/`：HAL、MUSE 和底盘协调器记录。
- `runtime/reports/`：依赖和离线测试报告。
- `models/policies/arm_moving_clock/`：包含 `model_7999.pt` 和实际使用的 `exported/policy.onnx`；单帧 60 维、历史 5 帧、12 维腿动作。

源码包包含源码、配置、必要模型和第三方许可证；不包含 `runtime/`、录制数据、缓存和本机环境。环境重建与验证步骤见 [开发说明](docs/development.md)。第三方组件归属和修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

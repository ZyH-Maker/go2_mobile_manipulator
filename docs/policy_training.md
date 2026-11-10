# 底盘策略来源、重新训练与部署适配

本文说明当前底盘策略与上游项目的关系，以及重新训练、台阶训练和更换模型时的修改范围。这是实施说明；不表示已经完成新策略训练或爬台阶真机验证。

## 1. LocoPilot 与训练仓库的关系

LocoPilot 直接把 `Giovadess/locomanipulation-teleop-isaaclab` 作为 Git 子模块，使用它的底盘强化学习训练和部署代码，不只是参考其方法。

| 项目 | 职责 |
|---|---|
| LocoPilot | 整合底盘、机械臂、双边遥操作、相机、数据采集与策略部署 |
| locomanipulation-teleop-isaaclab | Go2＋PiPER 的 Isaac Lab 强化学习环境、训练脚本、已有权重，以及 MuJoCo/ROS 2 部署控制器 |
| 本项目 | 将实际使用的组件整理为 Foxy 环境下的底盘入口、独立机械臂执行节点和通用轨迹接口 |

本项目保留的版本：

- LocoPilot：`93c48fc3ff12de4ee5b96b7d64644454ea7b0d48`。
- 强化学习子模块：`4f735d06848d3230055832f260d73e9f90133186`；上游子模块配置指定 `jazzy-deploy` 分支。
- 版本来源记录见 [source_versions.json](../third_party/source_versions.json)。本地训练与部署源码位于 [third_party/locomotion_policy](../third_party/locomotion_policy)。

### 论文与作者

两个仓库属于同一研究团队的相关工作，但核对版本的 README 引用不同论文。

LocoPilot 对应 **LocoPilot: Single-Operator Bilateral Teleoperation for Loco-Manipulation Policy Learning**，作者为 Antonio Langella、Giovanni Minelli、Giovanni B. Dessy、Lorenzo Amatucci、Giulio Turrisi、Victor Barasuol、Claudio Semini。该版本的引用条目年份为 2026，发表渠道和论文链接仍为 TODO，不能据此确认最终发表信息。

强化学习仓库注明维护者为 **Giulio Turrisi**，说明其受团队的 `trash-collection-isaaclab` 启发，并建议引用 **BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection**。作者包括 Giulio Turrisi、Angelo Bratta、Giovanni Minelli、Gabriel Fischer Abati、Amir H. Rad、João Carlos Virgolino Soares、Claudio Semini。因此不能将两个仓库简单等同为同一篇论文。

证据与参考：

- [LocoPilot README（锁定版本）](https://github.com/iit-DLSLab/LocoPilot/blob/93c48fc3ff12de4ee5b96b7d64644454ea7b0d48/README.md)
- [LocoPilot 子模块配置（锁定版本）](https://github.com/iit-DLSLab/LocoPilot/blob/93c48fc3ff12de4ee5b96b7d64644454ea7b0d48/.gitmodules)
- [强化学习仓库 README（锁定版本）](https://github.com/Giovadess/locomanipulation-teleop-isaaclab/blob/4f735d06848d3230055832f260d73e9f90133186/README.md)
- [原作者训练说明（锁定版本）](https://github.com/Giovadess/locomanipulation-teleop-isaaclab/blob/4f735d06848d3230055832f260d73e9f90133186/README_TRAIN.md)
- [BinWalker 论文](https://arxiv.org/abs/2603.10529)

## 2. 当前策略接口

当前使用 `models/policies/arm_moving_clock/`：

```text
单帧 60 维观测 × 历史 5 帧 → 300 维策略输入 → 12 维腿部动作
```

单帧观测包括底盘线速度、角速度、投影重力、速度指令、身体高度/俯仰指令、腿部关节位置偏差、腿部关节速度、上一时刻动作、机械臂六轴角度偏差和四维步态时钟。实际顺序以训练环境和部署封装为准。

- `model_7999.pt`：用于尝试恢复训练的检查点；需核对训练库、网络和检查点格式兼容性。
- `exported/policy.onnx`：当前部署推理使用的策略，不是恢复 PPO 训练的检查点。
- `params/env.yaml`、`params/agent.yaml`：与策略对应的环境及训练参数。
- 当前控制周期为 20 ms，即 50 Hz；训练配置为仿真步长 0.005 s、decimation 4。
- 默认策略腿部 PD 为 20/1.5，准备站立 PD 为 60/5。它们属于不同阶段。
- 策略只输出腿部动作。机械臂由独立执行节点控制，底盘读取真实机械臂角度。

当前部署入口仍固定使用 300 维预热输入和 20 ms 定时器，因此不是任意网络都能直接替换的通用加载器。

## 3. 哪些情况可以只更新策略

| 改动 | 是否能沿用当前部署 |
|---|---|
| 同接口重新训练权重 | 基本可以；更新策略、对应配置及必要归一化数据 |
| 保持接口，继续训练爬坡和上下台阶 | 可沿用部署框架；仍需仿真和真机验证 |
| 改 MLP 隐藏层宽度/深度，输入输出接口不变 | 导出兼容 ONNX 后通常可以；同时检查推理耗时 |
| 增加高度图、相机，改变历史长度或观测含义 | 修改部署端观测构造及传感器输入 |
| 改为带显式隐藏状态的循环网络 | 修改推理输入输出、隐藏状态维护及重置逻辑 |
| 改机器人结构、关节数量或动作类型 | 修改训练模型、关节映射和部署控制代码 |
| 策略同时输出机械臂动作 | 重新设计动作接口，与机械臂执行节点对接 |

即使输入输出维度相同，也必须匹配观测顺序、单位、坐标系、默认关节角、历史顺序、归一化、动作顺序、动作缩放、动作滤波、力矩限制、控制频率和 PD。不能仅凭 ONNX 成功加载就判断兼容。

修改网络结构后，原检查点通常不能直接完整恢复；可选择重新训练或实现兼容的部分权重迁移。PPO 恢复还涉及 critic 和优化器等状态，并非只检查 actor 输入输出。

## 4. 爬坡、上下台阶的训练方案

原仓库已经注册以下任务，拼写应原样保留：

```text
Locomomanipulation-Go2-Flat
Locomomanipulation-Go2-Rough-Blind
Locomomanipulation-Go2-Rough-Vision
```

当前 `arm_moving_clock/params/env.yaml` 已包含粗糙地形生成器：上下台阶高度范围 5～10 cm、台阶宽度 30 cm，上下台阶地形比例各 15%，另外包含平地、坡面和随机粗糙地形。这是保存配置所描述的训练设置，不等于已证明当前权重在本机上具有可靠的爬台阶能力。

建议先采用 Rough-Blind，保持当前 300 维输入和 12 维动作：

1. 匹配真实 Go2＋PiPER 的机器人模型，包含安装位置、质量、惯量和负载。
2. 按实际目标调整台阶高度、宽度和地形比例，由低台阶逐步增加难度。当前地形生成配置的 `curriculum=False`，需另行实现或配置难度递进。
3. 检查抬脚、碰撞、姿态、动作平滑与速度跟踪等奖励，以及摔倒等终止条件。
4. 若网络、观测和库版本兼容，加载现有训练检查点继续训练；否则重新训练或进行部分权重迁移。保留原策略用于对比。
5. 测试平地、上下台阶、停走、转向和不同机械臂姿态，记录成功率、摔倒、碰撞和动作变化。
6. 导出并完成部署适配及仿真验证，再进行真机测试。

Rough-Vision 会增加高度观测。仿真的高度扫描不能直接当作真实相机输入，部署时还需要实现高度图估计、坐标对齐和传感器数据接入，不能只更换权重。

如果 climb 指机械臂撑地、抓扶或协同攀爬，当前只控制腿部的 12 维策略不足以直接完成，需要新的任务和动作接口。

## 5. 更换机器人模型

需要区分“策略网络模型”和“机器人物理模型”。

原训练资产为 Go2＋**Piper L**。本项目中的标准 PiPER 限位 URDF 用于真实反馈检查，并不会自动把训练资产变为标准 PiPER。若实际使用标准版，应重新匹配臂长、质量、惯量、碰撞体、安装位姿和载荷。

机器人模型改变后，需要同步检查：

- Isaac Lab 使用的 USD 资产及执行器参数。
- 关节名称、顺序、零位、轴方向、默认站姿和动作尺度。
- 仿真中的机械臂运动范围和速度分布。
- 部署使用的 MuJoCo 模型、默认姿态、关节映射和力矩参数。

同样六个机械臂关节、同样十二个腿关节并不代表动力学相同。仅替换 URDF 或 ONNX 不能消除这些差异。

## 6. 训练环境与主要代码

上游锁定版本标注测试环境为 **Isaac Lab 2.3.2、rsl_rl 3.3.0、MuJoCo 3.7.0**。训练使用独立的 Isaac Lab 环境及适合的 NVIDIA GPU；当前 Foxy/Python 3.8 虚拟环境用于部署，不应直接当作训练环境。

下表路径相对于 `third_party/locomotion_policy/`；其中包根目录 `PKG` 指 `source/locomanipulation_teleop_isaaclab/locomanipulation_teleop_isaaclab/`。

| 路径 | 用途 |
|---|---|
| `PKG/tasks/locomanipulation/locomanipulation_env_cfg.py` | 地形、观测、奖励系数、控制周期和随机化 |
| `PKG/tasks/locomanipulation/locomanipulation_env.py` | 环境执行、动作处理、奖励与终止逻辑 |
| `PKG/assets/go2_piper_l_asset.py` 及对应 USD | 机器人结构、执行器和初始姿态 |
| `PKG/tasks/locomanipulation/agents/rsl_rl_ppo_cfg.py` | 网络、PPO 参数及训练次数 |
| `scripts/rsl_rl/train.py` | 训练和恢复检查点 |
| `scripts/rsl_rl/cli_args.py` | 恢复训练等命令行参数 |
| `scripts/rsl_rl/play.py` | 加载检查点、仿真测试及导出 ONNX/JIT |
| `deploy/locomotion_policy_wrapper.py` | 部署观测构造、历史缓存、动作转换 |
| `deploy/config.py` | 部署策略路径及配置加载 |

在已经安装好 Isaac Lab 的训练环境中，进入训练源码根目录执行：

```bash
python -m pip install -e source/locomanipulation_teleop_isaaclab

python scripts/rsl_rl/train.py \
  --task=Locomomanipulation-Go2-Rough-Blind \
  --num_envs=4096 \
  --headless
```

4096 是原作者示例的并行环境数，按显存调整。这是从头训练示例；恢复训练需根据 `cli_args.py` 的 `--resume`、`--load_run`、`--checkpoint` 参数及脚本的日志查找规则选择检查点，不能直接把 ONNX 路径作为训练检查点。

## 7. 新策略接入本项目

保持现有接口时，新增策略目录，保留当前已运行策略：

```text
models/policies/go2_piper_climb/
├── exported/
│   └── policy.onnx
└── params/
    ├── env.yaml
    └── agent.yaml
```

接入流程：

1. 使用训练环境的 `play.py` 加载检查点进行测试和导出；该脚本支持将策略归一化随网络导出。
2. 保存与新策略对应的环境、训练参数和来源信息。不要把旧 `env.yaml` 随意搭配新权重。
3. 修改 `third_party/locomotion_policy/deploy/config.py` 的 `locomotion_policy_folder_path` 指向新目录。
4. 对照第 3 节逐项确认接口；接口变化时同步修改 `locomotion/base_controller.py` 和策略封装。
5. 进行离线推理检查与适配后的 MuJoCo 验证，检查动作、关节方向、动力学和时序，再进行真机验证。现有离线测试通过不代表新策略已通过闭环验证。

接口保持一致时，机械臂执行节点、键盘控制和 Go2 准备→插值站立→人工启动策略的流程可以沿用，命令见 [项目 README](../README.md)。

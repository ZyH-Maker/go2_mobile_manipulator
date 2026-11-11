# Go2＋标准 PiPER：可指令身体姿态的腿部策略

这里是 `go2_mobile_manipulator` 的训练子项目，基于 Giovadess/locomanipulation-teleop-isaaclab 的 BSD-3-Clause 源码改造。主项目负责真机 ROS/CAN 接口；本目录负责机器人资产、强化学习训练、验证、导出和 MuJoCo sim2sim。

当前只注册两个任务：

| 任务 | 地形 |
|---|---|
| `Go2-Piper-Standard-Flat-v0` | 平面 |
| `Go2-Piper-Standard-Rough-v0` | 一个混合粗糙地形环境：平地、随机起伏、坡面、上下台阶，带地形课程 |

本版为 **Blind** 策略，无相机、图像网络、高度图 actor 输入或视觉蒸馏。仿真射线扫描只计算高度奖励和评估指标，不属于真机 actor 输入。台阶包含在 Rough 中，不额外注册任务。能力范围需要训练后测试，采样范围不等于真机许可范围。

## 1. 策略和控制结构

```text
速度＋身体指令 [vx, vy, wz, pitch, Δheight]
机器人本体状态＋真实机械臂角度＋历史
                      ↓
             腿部 PPO actor
                      ↓
             12 个腿部关节目标

独立轨迹生成器 → 标准 PiPER 六轴运动 → 给底盘产生姿态／动力学扰动
```

腿部策略不决定机械臂如何抓取，夹爪固定为 40 mm 开口。本次训练的是移动机械臂时的运动和平衡能力。后续高层可以同时发送五维身体指令、六轴机械臂目标和夹爪目标；抓取任务、视觉高层和夹爪执行接口尚不包含在本训练任务中。

统一契约位于 `policy_contract.py`：

- 物理步长 0.005 s，decimation 4，策略 50 Hz。
- 单帧 60 维，5 帧历史，actor 输入 `(1,300)`，输出 `(1,12)`。
- 单帧顺序：机身线速度 3、机身角速度 3、投影重力 3、速度指令 3、pitch/高度指令 2、腿部相对角度 12、腿速 12、前次原始动作 12、时钟 4、机械臂相对角度 6。
- 历史从旧到新；腿部顺序为 FL/FR/RL/RR 的 hip，然后 thigh，然后 calf。机械臂为 joint1～joint6。
- 默认腿姿态：hip=0、thigh=0.9、calf=-1.8 rad；默认臂姿态 `[0,1,-1.2,0,0.2,0]` rad。
- 动作裁剪 ±6，`0.8*当前＋0.2*前次` 滤波，乘 0.5 后加默认角。腿部 PD 20/1.5，髋／大腿限力矩 23.7 Nm、小腿 45.43 Nm；训练有 0～1 个物理步的执行延迟。
- 指令范围：vx ±0.5 m/s、vy ±0.25 m/s、wz ±0.5 rad/s、pitch ±0.25 rad、高度偏移 -0.08～0 m。各轴有变化率限制；整回合采样静止与移动姿态指令。
- **pitch 是相对世界水平面的绝对俯仰目标**，本版不再叠加地形俯仰；高度是相对局部扫描地面均值的 `0.30 m＋偏移`。
- 指令课程从 25% 范围逐渐扩大，默认 200000 个策略步达到全范围；PPO 每迭代 24 步，约 8333 迭代。

保持 300→12 只是保留接口形状，机器人、默认角、姿态语义和网络均已改变。**旧 Piper L 权重不能直接作为本版有效策略使用或恢复 PPO。**

## 2. 标准 PiPER 资产和碰撞

来源：主项目 `models/piper/piper.urdf` 与 `models/piper/meshes/`。生成器提取标准版关节变换、网格、质量和惯量，与原 Go2 资产组合成 `deploy/mujoco/models/go2_piper_standard.xml`；Isaac 由同一个 MJCF 转换 USD，减少两个仿真器手工维护的差异。

需要先核实的物理参数：

- 默认安装位置为机身坐标 `[0,0,0.07]` m，安装转角为零，是沿用参考安装方式，不是实测标定结果。修改安装位置后必须重新生成 MJCF、轨迹库和 USD，再训练。
- J6 采用当前真机执行器的 ±120° 限位，较来源 URDF 严格。其他轴来自标准版 URDF。
- 来源 URDF 两个夹指的 Iyy 为负。生成器按 20×60×10 mm 均匀盒重新计算这两个夹指惯量，保留原质量；修正写入 `.provenance.json`，未将其宣称为辨识值。
- 六轴机械臂采用位置伺服近似：200/10 PD、20 Nm 限力矩、目标速度不超过 25°/s、加速度不超过 50°/s²。它们是仿真起点，须用实测阶跃／轨迹响应校准，不能视为 PiPER 固件的真实电机参数。
- 末端独立 `payload` 惯性体随机质量 0.05～0.5 kg，并随机重心。它模拟携物负载，不包含抓取接触、负载拿起或放下事件。
- 本模型尚未加入用户设备的相机、支架及其质量。安装物实际存在时应在生成器中补齐，重新生成并训练。

启用臂—地面、臂—四足和非相邻连杆碰撞；只排除相邻／固定连接的碰撞对。每个臂连杆使用一个网格凸包，复杂凹形结构可能需要凸分解。Isaac 中臂、本体异常接触有惩罚并终止回合，同时检查失稳、离地高度和臂跟踪失败。净接触力阈值不是完整的碰撞距离证明，传感器与 USD 导入结果须验收。

`config/arm_motion_library.json` 是连通运动图。生成时检查每条边的多个中间点，以及名义腿姿态、pitch 0/±0.25、机身高度 0.22/0.30 m 下的 MuJoCo 碰撞。运行时随机沿图移动，使用五次平滑插值，按峰值速度／加速度计算时长；插值结束、实际臂关节距终点小于 2°后，才进入下一段。约一半轨迹到位后停留 0.3～1.5 s，其余连续衔接；到位等待超过 3 s 终止回合。

**运动图库是有限采样筛选，不是任意腿姿态、粗糙地形和外部障碍下的运行时无碰撞保证。** 真机仍需单独的轨迹检查和指令约束。本策略不能通过控制腿来保证一个任意机械臂命令必然可行。

## 3. 环境准备

在有 NVIDIA GPU 的训练机器上，按 Isaac Lab 官方说明安装 **Isaac Lab 2.3.2** 对应的 Isaac Sim 和 Python 环境，使用 **rsl-rl-lib 3.3.0**。不要把主项目 Foxy/Python 3.8 部署虚拟环境用于训练。PyTorch 版本按该 Isaac 环境安装，不要用工具依赖文件盲目升级。

以下命令中的 `python` 均指已经激活的 Isaac Lab 环境解释器；整个主项目应存在，不能只复制本目录而漏掉 `models/piper`：

```bash
cd /workspace/project/go2_mobile_manipulator/third_party/locomotion_policy
python -m pip install -r requirements-tools.txt
python -m pip install rsl-rl-lib==3.3.0
python -m pip install -e source/locomanipulation_teleop_isaaclab
```

MuJoCo 独立验证可以使用另一套 Python 环境，先安装 PyTorch，再安装 `requirements-tools.txt`。本地物理回归使用 MuJoCo 3.2.3；训练环境的 MJCF 导入与 PhysX 响应仍需单独验证。

## 4. 生成资产，先验收再训练

```bash
python scripts/assets/build_standard_model.py
# 如果实际安装位置不同，例如 x=0.02 m，替换为：
# python scripts/assets/build_standard_model.py --mount 0.02 0 0.07

python scripts/assets/build_arm_library.py --poses 32 --seed 42 --samples 25
python scripts/assets/convert_usd.py --headless
python scripts/list_envs.py

python scripts/check_isaac_env.py --task Go2-Piper-Standard-Flat-v0 --num-envs 8 --headless
python scripts/check_isaac_env.py --task Go2-Piper-Standard-Rough-v0 --num-envs 8 --headless
python -B -m unittest discover -s tests -v
```

生成的 USD 不提交 Git，位于资产目录的 `generated/`。训练会校验其来源 MJCF 哈希，旧 USD 会被拒绝。运动库也绑定 MJCF 哈希；更新资产必须重建。增加 `--poses` 扩大轨迹覆盖前，先检查碰撞形状和生成耗时。

Isaac smoke 检查 18 个主动关节、300 维观测、观测重复读取不改变状态、局部重置隔离以及物理步进有限性。零动作可能终止回合，不代表训练策略失败。若导入器改变／合并了 body 名称，载荷随机化或接触传感器会报错；应修正 USD 导入和 body 映射，不能通过关闭碰撞检查掩盖。

必须在 GUI 中确认：腿部和臂关节轴向／零位、安装方向、夹爪开口、碰撞凸包、自碰撞过滤、payload 所在位置；检查没有穿模、爆炸或连续无效重置。

## 5. 训练

先用少量环境跑通 PPO：

```bash
python scripts/rsl_rl/train.py --task Go2-Piper-Standard-Flat-v0 \
  --num_envs 16 --max_iterations 2 --headless
```

正式 Flat 和 Rough 都可以从头训练：

```bash
python scripts/rsl_rl/train.py --task Go2-Piper-Standard-Flat-v0 \
  --num_envs 2048 --seed 42 --headless

python scripts/rsl_rl/train.py --task Go2-Piper-Standard-Rough-v0 \
  --num_envs 2048 --seed 42 --headless
```

按显存减少环境数。默认本地 TensorBoard 日志，无需 WandB 登录：

```bash
tensorboard --logdir logs/rsl_rl
```

从新的 Flat 检查点继续训练 Rough 的例子：

```bash
python scripts/rsl_rl/train.py --task Go2-Piper-Standard-Rough-v0 \
  --num_envs 2048 --resume \
  --checkpoint /absolute/path/to/flat_run/model_9999.pt --headless
```

`--resume` 是开关，不写 `--resume True`。路径必须指向本版训练产生的 checkpoint，其相邻 `params/policy_contract.json` 必须匹配。恢复包括 PPO 状态；环境的随机采样进度和 curriculum 从新环境起步，不能当成逐位可复现的断点续跑。旧 Piper L checkpoint 会因缺少契约而拒绝；不实现静默部分加载。

主要观察：速度／yaw 跟踪、pitch／高度跟踪、异常接触、回合长度、arm 到位超时、Rough 地形级别。不以总 reward 上升单独判断成功。指令和地形课程可能需要按这些指标调慢。

## 6. Isaac 验证和导出

用明确 checkpoint 导出 JIT、ONNX 和配置：

```bash
python scripts/rsl_rl/play.py --task Go2-Piper-Standard-Flat-v0 \
  --checkpoint /absolute/path/to/flat_run/model_9999.pt \
  --num_envs 16 --steps 1000 --headless --report reports/isaac_flat.json
```

对 Rough 权重将任务改为 `Go2-Piper-Standard-Rough-v0`。`--steps 0` 只导出，不表示策略已验证。使用 `--video --video_length 500` 可录像；GUI 检查去掉 `--headless`。

分别验证指令，机械臂仍然运动：

```bash
python scripts/rsl_rl/play.py --task Go2-Piper-Standard-Flat-v0 \
  --checkpoint /absolute/path/to/flat_run/model_9999.pt \
  --num_envs 16 --steps 1000 --commands 0 0 0 0.2 -0.05 \
  --headless --report reports/isaac_pose.json
```

`--commands` 顺序为 vx、vy、wz、pitch、Δheight。外部指令经过限幅和变化率限制。评估关闭观测噪声，保留质量等环境随机化；报告记录 RMSE 和非时间上限终止次数。随机指令评估与固定目标评估都应做，多个种子和不同负载分别复测。

每个 run 的 `exported/` 包含：

```text
policy.pt                  # JIT actor，包含导出器提供的归一化
policy.onnx                # ONNX actor
policy_contract.json       # 模型哈希、单位、顺序、控制参数
params/                    # 原训练环境和 PPO 配置
model_*.pt                 # 对应训练检查点副本
```

数值验证：

```bash
python scripts/validate_export.py --policy /absolute/path/to/run/exported
```

该命令检查契约、输入输出、有限性、JIT/ONNX 随机输入数值一致性，生成 `export_validation.json`。它不验证平衡或台阶能力。

## 7. MuJoCo sim2sim

新入口不依赖 ROS，不读取旧 `deploy/config.py`，也不会默认加载旧权重：

```bash
python deploy/play_mujoco.py --policy /absolute/path/to/run/exported \
  --terrain flat --seconds 20 --commands 0 0 0 0.15 -0.05 --viewer

python deploy/play_mujoco.py --policy /absolute/path/to/run/exported \
  --terrain flat --seconds 20 --suite --report reports/mujoco_flat_suite.json

python deploy/play_mujoco.py --policy /absolute/path/to/rough_run/exported \
  --terrain rough --seconds 20 --commands 0.2 0 0 0 0 \
  --report reports/mujoco_rough_forward.json
```

默认每次先用零指令稳定 2 s，再平滑切换目标；机械臂持续沿同一个运动图运动。`--hold-arm` 固定初始臂姿态，可用于定位问题。`--payload 0.5` 测试负载，`--motor-delay 0/1` 测试执行延迟，`--seed` 控制随机轨迹。

Flat suite 包含原地、正负 pitch、降低高度、组合姿态、前进、后退、侧移、转向和移动中调整身体。报告记录跟踪 RMSE、碰撞对、失稳／到位超时与模拟时长。默认阈值只是初始验收建议：速度 RMSE <0.25 m/s、yaw <0.2 rad/s、pitch <0.12 rad、height <0.04 m、臂跟踪 <0.06 rad，并要求无异常接触／失稳。非零退出码表示未通过；保留失败报告用于定位，不应删除失败场景。

Rough MuJoCo 场景目前是 +x 方向固定的 5/7/9 cm 上下阶梯，初始位置为平地。它用于跨引擎检查，不等于 Isaac 随机 Rough 分布；不能把原地站立的 Rough suite 当作爬阶验收。应确认实际走上／下阶梯，进一步扩展地形、起点和方向。

无训练权重时可做模型 PD 冒烟：

```bash
python deploy/play_mujoco.py --model-only --seconds 20 --report reports/model_pd_smoke.json
```

该模式用 60/5 腿部站立 PD，不运行 RL，报告明确标记 `model_pd_smoke`；通过只能说明这组模型／轨迹能够完成该物理冒烟，不能证明学会了五维指令或越障。

## 8. 真机接入边界与旧文件清理

以下文件仍被主项目 `locomotion/base_controller.py` 使用，作为**旧 Piper L 权重兼容层**保留：`deploy/locomotion_policy_wrapper.py`、`deploy/config.py`、`deploy/mujoco/mujoco_utils.py`、旧 `scene_flat.xml`、`go2_piper_l.xml` 及引用的网格。`tasks/supervised_learning_networks.py` 是旧封装的导入依赖。本版训练不使用它们。

已移除旧训练 USD／Piper L 训练配置、Vision/RMA/并发估计训练分支、自定义旧执行器、调参模板、重复 ROS 控制器／仿真器／IK／相机示例和无用场景。原许可证与主项目旧权重保留。

**主项目真机入口仍运行旧权重，没有自动切换成本版。** 新权重部署必须接入新标准模型、`StandardPolicy` 的契约、五维指令、实际机械臂角度和现有 HAL 生命周期，完成部署时序／观测对齐及独立真机验收。不能只把新 ONNX 覆盖到旧策略目录。

## 9. 验证状态

本次修改在独立 Python 环境执行了模型加载、标准 URDF/MuJoCo 运动学对照、运动图路径碰撞回归、速度／加速度约束、局部重置隔离、目标到位逻辑、JIT/ONNX 导出数值一致性，以及 MuJoCo 模型 PD 冒烟。测试使用的临时 actor 是数值测试网络，不是训练成果。

结果：训练子项目 8 项测试通过，主项目 `locomotion/tests/test_base_core.py` 的 7 项测试通过；20 s 移动机械臂的 Flat 模型 PD 冒烟无异常接触或到位故障。保留的旧推理封装也完成模型加载与一次 12 维有限动作输出检查。

当前执行环境没有 Isaac Sim/Isaac Lab；**USD 转换、PhysX 接触／随机化、Isaac smoke、PPO 训练、训练后策略验证和真实策略 sim2sim 尚未运行**。上述步骤提供可执行入口，需在训练环境按顺序验收。没有生成或宣称已训练的新权重。

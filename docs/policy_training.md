# 底盘策略训练与部署

当前标准 PiPER 训练的唯一操作说明是 [训练子项目 README](../third_party/locomotion_policy/README.md)，包含机器人资产生成、碰撞与臂轨迹、Flat/Rough 训练、验证、导出和 MuJoCo sim2sim。

本次重构已替换旧训练环境，旧 `Locomomanipulation-Go2-*` 任务名不再注册。新任务为：

- `Go2-Piper-Standard-Flat-v0`
- `Go2-Piper-Standard-Rough-v0`

两者均为无视觉的 12 维腿部策略，跟踪 `vx、vy、wz、pitch、高度偏移`，在标准 PiPER 运动和负载扰动下训练。新策略需要重新训练；模型、默认姿态和 pitch 语义发生变化，不能把旧权重直接视作新任务策略。

## 与当前真机入口的关系

`models/policies/arm_moving_clock` 是保留的 Piper L 旧权重，仍供 `locomotion/base_controller.py` 使用。其推理封装、旧 MuJoCo 模型和网格是现有真机入口的依赖，清理时保留。新权重不能直接覆盖旧目录，需要接入新策略契约、标准 PiPER 模型和五维外部指令。

新训练没有在当前环境执行 PPO。已完成的本地数值／MuJoCo 检查与待完成的 Isaac 验收边界，见训练 README 的验证状态。

## 上游来源

- [locomanipulation-teleop-isaaclab](https://github.com/Giovadess/locomanipulation-teleop-isaaclab)：本地训练代码基础；锁定来源版本 `4f735d06848d3230055832f260d73e9f90133186`。
- [LocoPilot](https://github.com/iit-DLSLab/LocoPilot)：遥操作、相机和部署集成参考；锁定版本 `93c48fc3ff12de4ee5b96b7d64644454ea7b0d48`。
- [trash-collection-isaaclab](https://github.com/iit-DLSLab/trash-collection-isaaclab)：后续冻结腿部策略、训练操作高层的结构参考，使用 Aliengo＋Z1，未直接移植其任务或权重。

来源清单见 [source_versions.json](../third_party/source_versions.json)，许可证和本地改动见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

# 通用机械臂位置接口

只有 `arm/run_executor.py` 读写 CAN。两个键盘只生成 ROS 目标，新的 VLA/RL 节点可替换键盘。

| 接口 | 类型与含义 |
|---|---|
| `/follower/arm_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory`，六轴位置轨迹 |
| `/follower/feedback/joint_states` | `sensor_msgs/msg/JointState`，真实 q（rad），差分估计 dq |
| `/follower/arm_controller/status` | JSON String：ready、phase、error、actual_position、target_position |
| `/follower/arm_controller/{enable,recover,hold,stop}` | `std_srvs/srv/Trigger`，接管、恢复、保持、快速停止 |

目标必须包含 joint1～joint6 各一次；位置有限、单位 rad，velocity/acceleration/effort 留空。时间严格递增，零起始点只能等于当前指令，目标变化率上限 25°/s。支持未来时间戳；立即开始使用零 stamp，超过 0.1 秒过期拒绝。新轨迹从当前插值指令开始，完成保持终点。

enable/recover 成功响应只表示受理，等 ready=true 再发目标。轻微越界的 Enter 拒绝不会锁死 R；恢复最大每轴 5°、固件速度 5%、到位阈值 0.5°。普通跟踪差限制 2°，完整反馈期限 0.5 秒；启动后不自动发送控制指令。三个实际角度 CAN 帧均在 80 ms 内更新才发布角度。

执行中发现目标发布者退出或冲突，取消轨迹并保持实测位置。切换源先 hold，退出旧源，再运行新源。DDS 发现有延迟，没有额外目标心跳；在线但卡住的源可能让已接受轨迹执行到终点。非法新消息拒绝，原轨迹继续。

P/hold：取消并保持；接管前取消请求，已开始复位/使能但不能确认保持时快速停止并锁定。空格/stop 不等于断电失能，真正故障需重启执行节点。执行节点正常退出尝试保持，退出后不再刷新目标。

底盘读真实角度，超过 100 ms 缺失会停止。机械臂故障但仍有有效角度时，底盘不会仅凭 arm status 自动停止。两个系统的停止独立。

位置接口不能直接接力矩输出；笛卡尔模型需先做 IK。机械臂当前使用标准 Piper 限位，底盘 ONNX 仍是作者 Piper L 训练模型。

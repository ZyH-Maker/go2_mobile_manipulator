// Read-only until the coordinator confirms motion-controller release and arms us.
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <dls2_interface/msg/blind_state.hpp>
#include <dls2_interface/msg/imu.hpp>
#include <dls2_interface/msg/trajectory_generator.hpp>
#include "motor_crc.h"

class ManagedHal : public rclcpp::Node {
  using Clock = std::chrono::steady_clock;
  using Low = unitree_go::msg::LowState;
  using Traj = dls2_interface::msg::TrajectoryGenerator;
  const std::array<int, 12> map_{{3,4,5,0,1,2,9,10,11,6,7,8}};
  unitree_go::msg::LowCmd cmd_{};
  Clock::time_point state_time_{}, command_time_{};
  rclcpp::Subscription<unitree_go::msg::LowCmd>::SharedPtr command_monitor_;
  bool seen_state_=false, seen_command_=false, active_=false, fault_=false;
  std::string fault_reason_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_pub_;
  rclcpp::Publisher<dls2_interface::msg::BlindState>::SharedPtr blind_pub_;
  rclcpp::Publisher<dls2_interface::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
  rclcpp::Subscription<Low>::SharedPtr low_sub_;
  rclcpp::Subscription<Traj>::SharedPtr traj_sub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr arm_service_;
  rclcpp::TimerBase::SharedPtr timer_;
  double age(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
  bool fresh() {return seen_state_ && seen_command_ && age(state_time_)<0.1 && age(command_time_)<0.1;}
  void fail(const char* reason) {
    if (!fault_) {fault_reason_=reason; RCLCPP_ERROR(get_logger(), "%s；低层阻尼已锁定", reason);}
    fault_=true;
  }
public:
  ManagedHal() : Node("locopilot_managed_hal") {
    cmd_.head[0]=0xFE; cmd_.head[1]=0xEF; cmd_.level_flag=0xFF;
    for (auto &m : cmd_.motor_cmd) {m.mode=1; m.q=2.146E9f; m.dq=0; m.kp=0; m.kd=0; m.tau=0;}
    low_pub_=create_publisher<unitree_go::msg::LowCmd>("/lowcmd",1);
    command_monitor_=create_subscription<unitree_go::msg::LowCmd>("/lowcmd",rclcpp::SensorDataQoS(),
      [this](unitree_go::msg::LowCmd::ConstSharedPtr, const rclcpp::MessageInfo &info){
        if (*low_pub_ == info.get_rmw_message_info().publisher_gid) return;
        if(active_)fail("其他控制器正在实际发送 LowCmd");
      });
    blind_pub_=create_publisher<dls2_interface::msg::BlindState>("/blind_state",1);
    imu_pub_=create_publisher<dls2_interface::msg::Imu>("/imu",1);
    active_pub_=create_publisher<std_msgs::msg::Bool>("/locopilot/hal_active",1);
    low_sub_=create_subscription<Low>("/lowstate",rclcpp::SensorDataQoS(),[this](Low::SharedPtr s){
      bool valid=true;
      for (int i=0;i<12;++i) valid=valid && std::isfinite(s->motor_state[i].q) && std::isfinite(s->motor_state[i].dq);
      if (!valid) {if(active_)fail("电机反馈无效"); return;}
      seen_state_=true; state_time_=Clock::now();
      if (active_ && s->wireless_remote[0]==0x55 && s->wireless_remote[1]==0x51 && (s->wireless_remote[2]&0x30)) fail("遥控器 L2/R2 停止");
      dls2_interface::msg::BlindState b;
      b.joints_position.resize(12); b.joints_velocity.resize(12);
      b.joints_acceleration.resize(12); b.joints_effort.resize(12);
      for(int i=0;i<12;++i) {auto &m=s->motor_state[map_[i]]; b.joints_position[i]=m.q; b.joints_velocity[i]=m.dq; b.joints_acceleration[i]=m.ddq; b.joints_effort[i]=m.tau_est;}
      blind_pub_->publish(b);
      dls2_interface::msg::Imu imu;
      for(int i=0;i<3;++i) {imu.orientation[i]=s->imu_state.quaternion[i+1]; imu.angular_velocity[i]=s->imu_state.gyroscope[i]; imu.linear_acceleration[i]=s->imu_state.accelerometer[i];}
      imu.orientation[3]=s->imu_state.quaternion[0]; imu_pub_->publish(imu);
    });
    traj_sub_=create_subscription<Traj>("/trajectory_generator",1,[this](Traj::SharedPtr t){
      bool valid=t->joints_position.size()==12 && t->joints_velocity.size()==12 && t->kp.size()==12 && t->kd.size()==12;
      if(valid) for(int i=0;i<12;++i) valid=valid && std::isfinite(t->joints_position[i]) && std::isfinite(t->joints_velocity[i]) && std::isfinite(t->kp[i]) && std::isfinite(t->kd[i]) && std::abs(t->joints_position[i])<=4 && std::abs(t->joints_velocity[i])<=20 && t->kp[i]>=0 && t->kp[i]<=70 && t->kd[i]>=0 && t->kd[i]<=6;
      if(!valid) {if(active_)fail("腿部目标无效"); return;}
      if(fault_)return;
      for(int i=0;i<12;++i) {auto &m=cmd_.motor_cmd[map_[i]]; m.q=t->joints_position[i]; m.dq=t->joints_velocity[i]; m.kp=t->kp[i]; m.kd=t->kd[i]; m.tau=0;}
      command_time_=Clock::now(); seen_command_=true;
    });
    arm_service_=create_service<std_srvs::srv::SetBool>("/locopilot/arm_lowlevel",[this](std_srvs::srv::SetBool::Request::SharedPtr req,std_srvs::srv::SetBool::Response::SharedPtr res){
      if(!req->data) {if(active_)fail("协调器请求停止"); res->success=true; return;}
      // The coordinator has confirmed ReleaseMode/CheckMode. Like the original
      // HAL, do not impose a quiet period for commands sent before release:
      // doing so leaves Sport off while refusing the replacement controller.
      res->success=!fault_ && fresh() && count_publishers("/trajectory_generator")==1;
      res->message="低层输出已启用";
      if(!res->success) {
        res->message="拒绝接管：";
        if(fault_)res->message+="故障已锁定（"+fault_reason_+"）；";
        if(!seen_state_)res->message+="尚未收到腿部状态；";
        else if(age(state_time_)>=0.1)res->message+="腿部状态超时，数据年龄="+std::to_string(age(state_time_))+" 秒；";
        if(!seen_command_)res->message+="尚未收到腿部目标；";
        else if(age(command_time_)>=0.1)res->message+="腿部目标超时，数据年龄="+std::to_string(age(command_time_))+" 秒；";
        auto writers=count_publishers("/trajectory_generator");
        if(writers!=1)res->message+="腿部轨迹发布者数量="+std::to_string(writers)+"（要求为 1）；";
        RCLCPP_WARN(get_logger(),"%s",res->message.c_str());
      }
      if(res->success) {
        active_=true;
        RCLCPP_INFO(get_logger(),"HAL 已接管：按当前腿部目标开始低层控制");
      }
    });
    timer_=create_wall_timer(std::chrono::milliseconds(2),[this]{
      if(active_) {
        if(!fresh() || count_publishers("/trajectory_generator")!=1)fail("反馈或指令超时，或存在竞争发布者");
        if(fault_) for(int i=0;i<12;++i) {auto &m=cmd_.motor_cmd[i]; m.q=2.146E9f; m.dq=0; m.kp=0; m.kd=2; m.tau=0;}
        get_crc(cmd_); low_pub_->publish(cmd_);
      }
      // 50 Hz status; the control timer remains 500 Hz.
      static int divider=0;
      if(++divider==10) {divider=0; std_msgs::msg::Bool msg; msg.data=active_&&!fault_; active_pub_->publish(msg);}
    });
    RCLCPP_INFO(get_logger(),"等待接管：明确启用前不切换原控制器、不发送 LowCmd");
  }
};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<ManagedHal>());rclcpp::shutdown();}

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation

class RlGamesActor(nn.Sequential):
    def __init__(self, num_obs, num_actions):
        super(RlGamesActor, self).__init__(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions)
        )
        
class RlGamesAgent(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(RlGamesAgent, self).__init__()
        # state_dict의 키 이름과 정확히 일치하도록 멤버 변수 이름을 'actor'와 'critic'으로 짓습니다.
        self.actor = RlGamesActor(num_obs, num_actions)
        
        # critic도 state_dict에 있으므로, 로드 에러를 방지하기 위해 형태만 만들어줍니다.
        # nn.Sequential을 직접 멤버로 할당합니다.
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1) # Critic의 출력은 보통 1 (Value)
        )
        # 추론에 필요 없는 std 변수도 state_dict에 있으므로, 형태만 만들어줍니다.
        self.std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, obs):
        return self.actor(obs)
# ======================================================================


class RlPlayerNode(Node):
    def __init__(self):
        super().__init__('rl_player_node')
        
        package_path = get_package_share_directory('rne_sim2sim')
        model_path = os.path.join(package_path, 'models', 'trot_v0.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. 모델 로드 (상태 딕셔너리 전용) ---
        self.get_logger().info(f"Loading checkpoint dictionary from: {model_path}")

        try:
            # 1.1. 체크포인트 파일(딕셔너리) 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 1.2. 모델의 '껍데기'(구조) 생성
            # 관측/행동 공간의 크기를 정확히 알아야 합니다.
            num_observations = 48 # base(3+3+3) + commands(3) + joints(12+12+12)
            num_actions = 12
            
            agent_shell = RlGamesAgent(num_observations, num_actions).to(self.device)
            
            # 1.3. '내용물'(state_dict)을 '껍데기'에 채워넣기
            agent_shell.load_state_dict(checkpoint['model_state_dict'])
            
            # 1.4. 추론에 필요한 actor 네트워크만 self.model로 사용
            self.model = agent_shell.actor
            self.model.eval()
            self.get_logger().info("Successfully loaded model state_dict into the agent and set to eval mode.")

        except Exception as e:
            self.get_logger().fatal(f"Failed to load the model from state dictionary.")
            self.get_logger().fatal(f"This might be due to a mismatch in the model architecture (the Python class).")
            self.get_logger().fatal(f"Error details: {e}")
            rclpy.shutdown()
            return
        # --- 3. Isaac Lab 환경 파라미터 설정 (매우 중요!) ---
        self.joint_names = ['haa_1','hfe_1','kfe_1','haa_2','hfe_2','hke_2','haa_3','hfe_3','kfe_3','haa_4','hfe_4','kfe_4']
        self.num_joints = len(self.joint_names)
        
        # 3.1. 기본 자세 (Default Pose)
        # 이 값은 Isaac Sim의 로봇 설정 파일에서 찾은 실제 값으로 반드시 교체해야 합니다!
        # 아래는 일반적인 사족보행 로봇의 예시 값입니다. (순서는 self.joint_names와 일치해야 함)
        self.default_joint_pos = np.array([
            0.0, 0.0, 0.0, 0.0,    # HAA joints
            0.0, 0.0, 0.0, 0.0,    # HFE joints
            0.0, 0.0, 0.0, 0.0 # KFE joints
        ])
        # 위 값의 순서를 self.joint_names 순서에 맞게 재배열해야 할 수 있습니다.

        # 3.2. 액션 스케일 (Action Scale)
        self.action_scale = 0.2

        # 3.3. PD 제어 게인 (P/D Gains)
        # 이 값들은 로봇의 URDF/XACRO 파일의 <dynamics> 태그나 Isaac Sim 설정에서 찾아야 합니다.
        self.p_gains = np.full(self.num_joints, 30)  # 예시: Stiffness
        self.d_gains = np.full(self.num_joints, 0.02)   # 예시: Damping

        # --- 4. ROS 2 인터페이스 설정 ---
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.action_pub = self.create_publisher(Float64MultiArray, '/joint_effort_controller/commands', 10)

        # --- 5. 상태 변수 초기화 ---
        self.joint_pos = None
        self.joint_vel = None
        self.base_lin_vel = None
        self.base_ang_vel = None
        self.projected_gravity = None
        
        self.velocity_commands = np.array([0.5, 0.0, 0.0]) # 명령: 앞으로 0.5m/s 직진
        self.last_action = np.zeros(self.num_joints)
        
        # --- 6. 타이머 시작 ---
        self.timer = self.create_timer(0.02, self.inference_loop) # 50 Hz
        self.get_logger().info("RL Player node has been initialized successfully.")

    def odom_callback(self, msg: Odometry):
        world_lin_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        world_ang_vel = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        orientation_q = msg.pose.pose.orientation
        quat = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        rot = Rotation.from_quat(quat)
        self.base_lin_vel = rot.apply(world_lin_vel, inverse=True)
        self.base_ang_vel = rot.apply(world_ang_vel, inverse=True)
        world_gravity_vector = np.array([0, 0, -1.0])
        self.projected_gravity = rot.apply(world_gravity_vector, inverse=True)

    def joint_state_callback(self, msg: JointState):
        ordered_pos = np.zeros(self.num_joints)
        ordered_vel = np.zeros(self.num_joints)
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.name.index(name)
                ordered_pos[i] = msg.position[idx]
                ordered_vel[i] = msg.velocity[idx]
            except ValueError:
                pass
        self.joint_pos = ordered_pos
        self.joint_vel = ordered_vel

    def inference_loop(self):
        # 모든 관측 데이터가 수신될 때까지 대기
        if any(v is None for v in [self.joint_pos, self.joint_vel, self.base_lin_vel]):
            self.get_logger().warn("Waiting for observation data...", throttle_duration_sec=2)
            return

        # 1. 관측 벡터 생성 (Isaac Lab 명세와 동일)
        relative_joint_pos = self.joint_pos - self.default_joint_pos
        
        observation = np.concatenate([
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.velocity_commands,
            relative_joint_pos,
            self.joint_vel,
            self.last_action,
        ]).astype(np.float32)

        # 2. 모델 추론
        obs_tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        
        # 모델의 출력은 -1~1 범위의 정규화된 값
        normalized_action = action_tensor.squeeze(0).cpu().numpy()
        
        # 다음 스텝의 관측을 위해 저장
        self.last_action = normalized_action.copy()
        
        # 3. 목표 위치 계산 (Isaac Lab 방식)
        target_position = (normalized_action * self.action_scale) + self.default_joint_pos
        # 4. PD 제어로 최종 토크(Torque) 계산
        target_velocity = np.zeros(self.num_joints)
        position_error = target_position - self.joint_pose
        velocity_error = target_velocity - self.joint_vel
        torques = self.p_gains * position_error + self.d_gains * velocity_error
        
        # 5. 계산된 토크 값을 Gazebo로 전송
        action_msg = Float64MultiArray()
        action_msg.data = torques.tolist()
        self.action_pub.publish(action_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RlPlayerNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
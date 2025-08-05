#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation

class RlPlayerNode(Node):
    def __init__(self):
        super().__init__('rl_player_node')
        
        package_path = get_package_share_directory('rne_sim2sim_py')
        model_path = os.path.join(package_path, 'models', 'trot_v0.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.get_logger().info(f"RL model loaded from {model_path}")
        
        self.joint_names = ['haa_3','haa_4','haa_1','haa_2','hfe_1','hfe_2','hfe_3','hfe_4','kfe_4','kfe_3','kfe_1','kfe_2']
        self.num_joints = len(self.joint_names)
        
        self.default_joint_pos = np.zeros(self.num_joints) # 실제 값으로 교체 필요!

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom/data', self.odom_callback, 10)
        self.action_pub = self.create_publisher(Float64MultiArray, '/joint_effort_controller/commands', 10)

        self.base_lin_vel = None
        self.base_ang_vel = None
        self.projected_gravity = None
        self.joint_pos = None
        self.joint_vel = None
        
        self.velocity_commands = np.array([0.5, 0.0, 0.0])
        self.last_action = np.zeros(self.num_joints)

        self.p_gains = np.full(self.num_joints, 30.0)  # Proportional gains
        self.d_gains = np.full(self.num_joints, 0.02)   # Derivative gains
        
        self.timer = self.create_timer(0.02, self.inference_loop)

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
        if self.joint_pos is None or self.base_lin_vel is None:
            self.get_logger().warn("Waiting for observation data...", throttle_duration_sec=2)
            return

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

        obs_tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        
        target_position = action_tensor.squeeze(0).cpu().numpy()
        self.last_action = target_position.copy()
        
         # === PD 제어 로직으로 최종 토크(Torque) 계산 ===
        # PD 제어 공식: Torque = P * (목표 위치 - 현재 위치) + D * (목표 속도 - 현재 속도)
        
        # 목표 속도(target_velocity)는 0으로 가정합니다 (안정적으로 서있거나 움직이는 상태 유지).
        target_velocity = np.zeros(self.num_joints)

        # 위치 오차 (Position Error)
        position_error = target_position - self.joint_pos

        # 속도 오차 (Velocity Error)
        velocity_error = target_velocity - self.joint_vel

        # 최종 토크 계산
        torques = self.p_gains * position_error + self.d_gains * velocity_error
        # ==============================================

        # 계산된 토크 값을 Float64MultiArray 메시지에 담아 발행
        action_msg = Float64MultiArray()
        action_msg.data = torques.tolist()
        self.action_pub.publish(action_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RlPlayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# 이전에 제안했던 가장 간단한 버전으로 돌아갑니다.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'rne_sim2sim'
    robot_xacro_file = 'cerberus.xacro'
    world_file = 'my_world.sdf'

    pkg_path = get_package_share_directory(pkg_name)
    world_path = os.path.join(pkg_path, 'worlds', world_file)
    xacro_file_path = os.path.join(pkg_path, 'urdf', robot_xacro_file)

    robot_description_config = xacro.process_file(xacro_file_path)
    robot_description = {'robot_description': robot_description_config.toxml()}

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-string', robot_description_config.toxml(),
            '-name', 'my_robot',
            '-allow_renaming', 'true'
        ],
    )
    
    return LaunchDescription([
        start_gazebo_server,
        robot_state_publisher_node,
        spawn_entity_node,
    ])  
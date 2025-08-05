import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('rne_sim2sim')
    
    xacro_path = os.path.join(pkg_path, 'urdf', 'cerberus.xacro')
    controllers_file = os.path.join(pkg_path, 'config', 'controllers.yaml')
    world_path = os.path.join(pkg_path, 'worlds', 'my_world.sdf')

    robot_urdf = xacro.process_file(xacro_path)
    robot_description = {'robot_description': robot_urdf.toxml()}

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    robot_state_publisher = Node(
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
            '-string', robot_urdf.toxml(),
            '-name', 'my_quadruped',
            '-allow_renaming', 'true',
            '-z', '0.5'
        ],
    )
    
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description],
        output='screen'
    )
    
    rl_player_node = Node(
        package='rne_sim2sim_py',
        executable='rl_player',
        name='rl_player_node',
        output='screen'
    )
    
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    
    joint_effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_effort_controller', '--controller-manager', '/controller_manager'],
    )

    spawn_to_broadcaster_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    
    broadcaster_to_controller_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_effort_controller_spawner],
        )
    )

    controller_to_rl_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_effort_controller_spawner,
            on_exit=[rl_player_node],
        )
    )

    return LaunchDescription([
        start_gazebo_server,
        robot_state_publisher,
        spawn_entity_node,
        control_node,
        spawn_to_broadcaster_handler,
        broadcaster_to_controller_handler,
        controller_to_rl_handler,
    ])
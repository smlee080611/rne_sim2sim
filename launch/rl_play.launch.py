import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # 1. 경로 설정
    # 패키지 이름이 rne_sim2sim이 맞는지 확인
    pkg_path = get_package_share_directory('rne_sim2sim') 
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # 2. XACRO 파일 파싱
    # 메인 xacro 파일 이름을 확인 (예: cerberus.xacro)
    xacro_file = os.path.join(pkg_path, 'urdf', 'cerberus.xacro') 
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    robot_description = {'robot_description': robot_description_raw}
    
    # 3. 월드 파일 경로 설정
    world_file = os.path.join(pkg_path, 'worlds', 'my_world.world')

    # 4. Gazebo Classic 실행
    # 4.1. 서버(물리 엔진) 실행
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'true'}.items()
    )

    # 4.2. 클라이언트(GUI) 실행
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # 5. 로봇 상태 발행 노드
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # 6. 로봇 스폰 노드
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_quadruped',
                   '-x','0.0',
                   '-y','0.0',
                   '-z','0.27'],#0.27
        output='screen'
    )

    # 7. 컨트롤러 로드 (Spawner)
    # Spawner들은 Gazebo 내부의 gazebo_ros2_control 플러그인이 제공하는
    # /controller_manager 서비스를 찾아서 컨트롤러를 로드합니다.
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "300"],
    )
    joint_effort_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_effort_controller", "--controller-manager-timeout", "300"],
    )
    
    # 8. RL 추론 노드
    rl_player_node = Node(
        package='rne_sim2sim', # 패키지 이름 확인
        executable='rl_player',
        name='rl_player_node',
        output='screen'
    )

    # 실행할 모든 노드를 리스트에 담아 반환
    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher_node,
        spawn_entity_node,
        joint_state_broadcaster_spawner,
        joint_effort_controller_spawner,
        rl_player_node,
    ])
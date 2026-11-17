###############################################################################################
# TM5-700, project gripper, wrist D435i, TM driver, MoveIt, and RViz.
###############################################################################################

import fcntl
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import xacro
import yaml


_RUNTIME_LOCK_FILE = None


def acquire_runtime_lock(lock_path="/tmp/book_grasp_tm5_moveit.lock"):
    """Prevent two copies of this combined MoveIt launch in one host session."""

    global _RUNTIME_LOCK_FILE
    # Open without truncating: a competing launch must not erase the current
    # owner's diagnostics before its non-blocking flock attempt fails.
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.seek(0)
        holder = lock_file.read().strip() or "holder metadata unavailable"
        lock_file.close()
        raise RuntimeError(
            "Another TM5 book-grasp MoveIt launch is already running. Stop it "
            "before starting online or offline mode. "
            f"Lock holder: {holder}"
        ) from error

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(
        f"pid={os.getpid()}\n"
        f"ros_domain_id={os.environ.get('ROS_DOMAIN_ID', '0')}\n"
        "mode=tm5_combined_moveit_launch\n"
    )
    lock_file.flush()
    _RUNTIME_LOCK_FILE = lock_file


def launch_after_preflight(event, _context, runtime_nodes):
    """Start the runtime only after the ROS graph is confirmed conflict-free."""

    if event.returncode != 0:
        return [
            EmitEvent(
                event=Shutdown(
                    reason=(
                        "Existing MoveIt/TM driver detected. Refusing to mix online "
                        "and offline planning states in one ROS domain."
                    )
                )
            )
        ]
    return runtime_nodes


def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError:
        return None


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def build_planning_pipeline_config(moveit_config_package):
    """Load OMPL for interactive RViz and explicit Pilz PTP for the coordinator."""
    config = {
        # RViz sends an empty planner_id when an operator drags an interactive
        # marker without selecting a planner. Pilz requires an explicit
        # PTP/LIN/CIRC planner_id and rejects that request before IK or
        # collision checking. Keep OMPL as the interactive default. The grasp
        # coordinator explicitly selects Pilz/PTP for production PTP requests.
        "planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        },
    }
    config["ompl"].update(
        load_yaml(moveit_config_package, "config/ompl_planning.yaml")
    )
    pilz_planning = load_yaml(
        moveit_config_package,
        "config/pilz_industrial_motion_planner_planning.yaml",
    )
    # Sequence capabilities are move_group-wide plugins, not parameters of
    # the planning pipeline.  PTP planning itself only needs CommandPlanner.
    pilz_planning.pop("capabilities", None)
    config["pilz_industrial_motion_planner"] = pilz_planning
    return config


def generate_launch_description():
    acquire_runtime_lock()
    robot_ip_argument = DeclareLaunchArgument(
        "robot_ip",
        default_value="",
        description="IPv4 address; empty means isolated no-robot visualisation mode",
    )
    robot_ip = LaunchConfiguration("robot_ip")
    offline_joint_state_topic_argument = DeclareLaunchArgument(
        "offline_joint_state_topic",
        default_value="/tm5_offline/joint_states",
        description="Joint-state topic used only when robot_ip is empty",
    )
    offline_joint_state_topic = LaunchConfiguration("offline_joint_state_topic")
    raw_joint_state_topic_argument = DeclareLaunchArgument(
        "raw_joint_state_topic",
        default_value="/tm5_driver/joint_states",
        description="Unmodified TM driver state topic; used only with robot_ip",
    )
    raw_joint_state_topic = LaunchConfiguration("raw_joint_state_topic")
    gripper_joint_state_topic_argument = DeclareLaunchArgument(
        "gripper_joint_state_topic",
        default_value="/gripper/joint_states",
        description=(
            "Measured gripper JointState topic. It may be absent while the gripper node is off; "
            "the motor must then remain unknown."
        ),
    )
    gripper_joint_state_topic = LaunchConfiguration("gripper_joint_state_topic")
    use_rviz_argument = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz; set false for headless launch smoke tests",
    )
    use_rviz = LaunchConfiguration("use_rviz")
    has_robot_ip = PythonExpression(["bool('", robot_ip, "')"])
    joint_state_topic = PythonExpression(
        [
            "'/joint_states' if bool('",
            robot_ip,
            "') else '",
            offline_joint_state_topic,
            "'",
        ]
    )
    trajectory_execution_enabled = ParameterValue(
        PythonExpression(["bool('", robot_ip, "')"]), value_type=bool
    )

    description_package = "tm5_manual_book_grasp"
    moveit_config_package = "tm5-700_moveit_config"

    robot_description_xml = xacro.process_file(
        os.path.join(
            get_package_share_directory(description_package),
            "urdf",
            "tm5-700_with_gripper.xacro",
        )
    ).toxml()
    robot_description = {"robot_description": robot_description_xml}

    robot_description_semantic = {
        "robot_description_semantic": load_file(
            description_package, "config/tm5-700_with_gripper.srdf"
        )
    }
    robot_description_kinematics = {
        "robot_description_kinematics": load_yaml(
            moveit_config_package, "config/kinematics.yaml"
        )
    }

    ompl_planning_pipeline_config = build_planning_pipeline_config(
        moveit_config_package
    )

    controllers_yaml = load_yaml(moveit_config_package, "config/controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml,
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
    }
    trajectory_execution = {
        "moveit_manage_controllers": trajectory_execution_enabled,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.1,
        # MoveIt 2.5.9 reads "execution_duration_monitoring".  The older
        # "trajectory_duration_monitoring" spelling in TM's YAML is ignored
        # by this installed MoveIt version.  TM's action result waits for the
        # QueueTag reply, which arrives after the nominal PVT duration, so the
        # generic duration watchdog would otherwise cancel a valid goal.
        "trajectory_execution.execution_duration_monitoring": False,
    }
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    robot_description_planning = load_yaml(
        description_package, "config/tm5-700_joint_limits.yaml"
    )
    robot_description_planning.update(
        load_yaml(moveit_config_package, "config/pilz_cartesian_limits.yaml")
    )
    joint_limits = {"robot_description_planning": robot_description_planning}

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        emulate_tty=True,
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            joint_limits,
            {"use_sim_time": False},
            {"allow_trajectory_execution": trajectory_execution_enabled},
        ],
        remappings=[("joint_states", joint_state_topic)],
    )

    rviz_config_file = os.path.join(
        get_package_share_directory(description_package),
        "config",
        "run_move_group.rviz",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        emulate_tty=True,
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            robot_description_kinematics,
            joint_limits,
        ],
        remappings=[("joint_states", joint_state_topic)],
    )

    world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base"],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        remappings=[("joint_states", joint_state_topic)],
    )
    offline_joint_state_publisher = Node(
        package="tm5_manual_book_grasp",
        executable="offline_joint_state_publisher.py",
        name="tm5_offline_joint_state_publisher",
        output="screen",
        condition=UnlessCondition(has_robot_ip),
        remappings=[("joint_states", offline_joint_state_topic)],
    )
    tm_joint_state_adapter = Node(
        package="tm5_manual_book_grasp",
        executable="tm_joint_state_adapter.py",
        name="tm_joint_state_adapter",
        output="screen",
        condition=IfCondition(has_robot_ip),
        remappings=[
            ("input_joint_states", raw_joint_state_topic),
            ("gripper_joint_states", gripper_joint_state_topic),
            ("joint_states", joint_state_topic),
        ],
    )
    tm_driver = Node(
        package="tm_driver",
        executable="tm_driver",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(has_robot_ip),
        arguments=[["robot_ip:=", robot_ip]],
        remappings=[("joint_states", raw_joint_state_topic)],
    )

    graph_preflight = Node(
        package="tm5_manual_book_grasp",
        executable="tm_moveit_graph_preflight.py",
        name="tm_moveit_graph_preflight",
        output="screen",
        parameters=[{"discovery_timeout_sec": 1.5}],
    )

    runtime_nodes = [
        tm_driver,
        offline_joint_state_publisher,
        tm_joint_state_adapter,
        rviz,
        world_to_base,
        robot_state_publisher,
        move_group,
    ]
    start_runtime_after_preflight = RegisterEventHandler(
        OnProcessExit(
            target_action=graph_preflight,
            on_exit=lambda event, context: launch_after_preflight(
                event, context, runtime_nodes
            ),
        )
    )

    return LaunchDescription(
        [
            robot_ip_argument,
            offline_joint_state_topic_argument,
            raw_joint_state_topic_argument,
            gripper_joint_state_topic_argument,
            use_rviz_argument,
            start_runtime_after_preflight,
            graph_preflight,
        ]
    )

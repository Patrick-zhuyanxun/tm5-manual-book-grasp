cmake_minimum_required(VERSION 3.8)
project(tm5_manual_book_grasp)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf gripper_urdf config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/manual_book_grasp.py
  scripts/gripper_node.py
  scripts/offline_joint_state_publisher.py
  scripts/tm_joint_state_adapter.py
  scripts/tm_moveit_graph_preflight.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/aruco_geometry.py
  scripts/manual_target.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

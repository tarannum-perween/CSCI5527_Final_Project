cmake_minimum_required(VERSION 3.8)
project(scene_classifier)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scene_classifier/scene_classifier_node.py
  scene_classifier/nav2_context_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

cmake_minimum_required(VERSION 3.0.2)
project(lidar_ndt_localization)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
)
find_package(OpenMP REQUIRED)
message(STATUS "NDT_OMP_LIBRARIES: ${NDT_OMP_LIBRARIES}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "catkin_LIBRARIES: ${catkin_LIBRARIES}")

find_package(PCL REQUIRED COMPONENTS common io visualization filters search)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(lidar_localization_node src/localization.cpp)
target_link_libraries(lidar_localization_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
   OpenMP::OpenMP_CXX
)

install(TARGETS lidar_localization_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


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
  ndt_omp
)

message(STATUS "NDT_OMP_LIBRARIES: ${NDT_OMP_LIBRARIES}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "catkin_LIBRARIES: ${catkin_LIBRARIES}")

find_package(PCL REQUIRED COMPONENTS common io visualization filters search)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${NDT_OMP_INCLUDE_DIRS}
)

add_executable(lidar_localization_node src/localization.cpp src/ukf_ctrv.cpp)
target_link_libraries(lidar_localization_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${NDT_OMP_LIBRARIES}  # ensure ndt_omp is explicitly here
)

install(TARGETS lidar_localization_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(ndt_scan_align src/testing_scan_match.cpp)
target_link_libraries(ndt_scan_align
${NDT_OMP_LIBRARIES}  # link ndt_omp first
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)

install(TARGETS ndt_scan_align
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

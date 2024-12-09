cmake_minimum_required(VERSION 3.0.2)
project(lidar_ndt_localization)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

# Find catkin and any catkin packages
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

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
else()
  message(FATAL_ERROR "OpenMP not found")
endif()

# Find PCL (Point Cloud Library)
find_package(PCL REQUIRED COMPONENTS common io visualization filters search features registration)

message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "catkin_LIBRARIES: ${catkin_LIBRARIES}")

# Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs pcl_ros
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Link PCL libraries
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(lidar_localization_node src/localization.cpp)
target_link_libraries(lidar_localization_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
add_dependencies(lidar_localization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(map_indexer_node src/MapIndexer.cpp)
target_link_libraries(map_indexer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
add_dependencies(map_indexer_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


install(TARGETS 
  lidar_localization_node
  map_indexer_node
  # Add other executables here if needed
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Optionally install launch files, configuration files, etc.
# install(DIRECTORY launch config
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

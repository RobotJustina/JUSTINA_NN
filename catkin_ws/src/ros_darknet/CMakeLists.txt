cmake_minimum_required(VERSION 2.8.3)
project(yolo)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")
message(STATUS "CMAKE Module Path at ${CMAKE_MODULE_PATH}")



add_definitions(-std=c++11)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	geometry_msgs
	#justina_tools
	roslib
	sensor_msgs
	std_msgs
	tf
	#vision_msgs
	cv_bridge
	visualization_msgs
	actionlib
	image_transport
	)

find_package(Eigen3 REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

set(DARKNET_DIR "/home/biorobotica/darknet")

if(DARKNET_DIR)               # <--- Use 'BOOST_DIR', not 'DEFINED ${BOOST_DIR}'
	find_package(CUDA REQUIRED)
	#find_package(gflags REQUIRED)

	catkin_package(
		#  INCLUDE_DIRS include
		#  LIBRARIES openpose
		#  CATKIN_DEPENDS roscpp rospy std_msgs
		#  DEPENDS system_lib
		# DEPENDS Eigen3
		)

	include_directories(
		include
		${catkin_INCLUDE_DIRS}
		${CUDA_INCLUDE_DIRS}
		${Eigen_INCLUDE_DIRS}
		${DARKNET_DIR}/include
                ${DARKNET_DIR}/src
                ${OpenCV_LIBS}
		)

	add_library(Demo src/demo.cpp src/image_interface.c)
	target_link_libraries(Demo 
		${catkin_LIBRARIES} 
		${Eigen_LIBRARIES}
		${OpenCV_LIBRARIES}
		${DARKNET_DIR}/libdarknet.so
		)

	#add_executable(open_pose_node src/open_pose_node.cpp)
	#target_link_libraries(open_pose_node ${catkin_LIBRARIES} OpenPose)
	add_executable(darknet src/yolo_node.cpp)
	target_link_libraries(darknet
		${catkin_LIBRARIES}
		Demo
	)
else()
	message(WARNING "Undefined DARKNET_HOME env var, this package will not be compiled.")
endif()


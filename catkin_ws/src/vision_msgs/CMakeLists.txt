cmake_minimum_required(VERSION 2.8.3)
project(vision_msgs)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
	FILES
	VisionFaceObject.msg
	VisionFaceObjects.msg
	OpenPoseKeyPoint.msg
	OpenPoseRecognition.msg
	OpenPoseRecognitions.msg
	MaskBoundingBox.msg
	MaskBoundingBoxes.msg
)

## Generate services in the 'srv' folder
add_service_files(
	FILES
	FaceRecognition.srv
	OpenPoseRecognize.srv
	ObjectRecognize.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
	DEPENDENCIES
	std_msgs sensor_msgs geometry_msgs
)

###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES facenet
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

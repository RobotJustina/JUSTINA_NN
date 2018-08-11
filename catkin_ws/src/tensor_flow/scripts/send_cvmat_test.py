#!/usr/bin/env python
import sys
import rospy
import cv2 
from tensor_flow.srv import *
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError

if __name__=="__main__":
    bridge = CvBridge()
    image_decoded = cv2.imread("/home/biorobotica/docker_volumen/JUSTINA/catkin_ws/src/tensor_flow/scripts/diente-deleon.jpg")
    image_message = bridge.cv2_to_imgmsg(image_decoded, encoding="bgr8")
    rospy.wait_for_service('/tensor_flow/image')
    try:
        image_srv = rospy.ServiceProxy('/tensor_flow/image', image_srv)
        label = image_srv(image_message)
    except rospy.ServiceException, e:
        print ("Service call failed: %s" %e)

    print(label)

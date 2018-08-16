#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import roslib
roslib.load_manifest('facenet')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import tensorflow as tf
import align.detect_face
import numpy as np

def facenet_align(cv_image):
    print("facnet align function")
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7] # three step's threshold
    factor = 0.709 # scale factor

    imarray = np.asarray(cv_image)
    
    #with tf.Graph().as_default():
    #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
     #       pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None) 
                        
    bounding_boxes, _ = align.detect_face.detect_face(imarray, minsize, pnet, rnet, onet, threshold, factor)


def imageCallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    facenet_align(cv_image)
            
    cv2.imshow("Image", cv_image)
    cv2.waitKey(3)

def main(args):
    rospy.init_node('image_converter', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw", Image, imageCallback)
    rate = rospy.Rate(3) #30Hz

    global bridge
    global gpu_memory_fraction
    global pnet
    global rnet
    global onet

    bridge = CvBridge()
    gpu_memory_fraction = args.gpu_memory_fraction
        
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None) 
     
    while not rospy.is_shutdown():
        rate.sleep()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

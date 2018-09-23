#!/usr/bin/env python

import sys
import rospy
import rosgraph
import cv2
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from facenet_ros.msg import *
from facenet_ros.srv import *

from scipy import misc
import os
import re
import argparse
import time
import tensorflow as tf
import align.detect_face
import numpy as np
import math
import pickle
import facenet

import numpy as np
import face_recognition as face
import tensorflow as tf

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 0, 0), 2)
            if face.name is not None:
                if face.probability > threshold_reco:
                    cv2.putText(frame, face.name, (face_bb[0], face_bb[3] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=1, lineType=1)
                    cv2.putText(frame, str(face.probability.round(decimals=2)), (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=1, lineType=1)
                else:
                    cv2.putText(frame, "Unknown", (face_bb[0], face_bb[3] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=1, lineType=1)

def face_recognition_callback(req):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    with facenetGraph.as_default():
        with face_recognition.encoder.sess.as_default():
            try:
                cv_image = bridge.imgmsg_to_cv2(req.imageBGR, "bgr8")
            except CvBridgeError as e:
                print(e)
            faces = face_recognition.identify(cv_image)
    
    recog_faces = [] 
    for face in faces:
        if(req.id == '' or face.name == req.id.replace("_"," ")):
            face_name = face.name
            confidence = face.probability
            if(face.probability <= threshold_reco and req.id == ''):
                face_name = "Unknown"
            elif(req.id != '' and face.probability <= threshold_reco):
                continue
                
            bounding_box = [Point(face.bounding_box[0], face.bounding_box[1], 0), Point(face.bounding_box[2], face.bounding_box[3], 0)]
            face_centroid = Point((face.bounding_box[0] + face.bounding_box[2]) / 2, (face.bounding_box[1] + face.bounding_box[3] / 2), 0)
            face_class = VisionFaceObject(id=face_name, confidence=confidence, face_centroid=face_centroid, bounding_box=bounding_box)
            recog_faces.append(face_class);

    return FaceRecognitionResponse(VisionFaceObjects(h, recog_faces))

def main(args):
    global bridge
    global facenetGraph
    global face_recognition
    global threshold_reco
    global s

    initRos = False
     
    model_file = args.model_file
    classifier_file = args.classifier_file
    image_size = args.image_size
    margin = args.margin
    gpu_memory_fraction = args.gpu_memory_fraction
    detect_multiple_faces = args.detect_multiple_faces
    threshold_reco = args.threshold_reco

    bridge = CvBridge()

    facenetGraph = tf.Graph()
    with facenetGraph.as_default():
        face_recognition = face.Recognition(facenet_model=model_file, classifier_model=classifier_file, face_crop_size=image_size, threshold=[0.7,0.8,0.8], factor=0.709, face_crop_margin=margin, gpu_memory_fraction=gpu_memory_fraction, detect_multiple_faces=detect_multiple_faces)

    while 1:
        if rospy.is_shutdown():
            break;
        if rosgraph.is_master_online():
            if not initRos:
                print 'Creating the ros node and service'
                rospy.init_node('facenet_node', anonymous=True)
                rate = rospy.Rate(30) #30Hz
                s = rospy.Service('/vision/facenet_recognizer/faces', FaceRecognition, face_recognition_callback)
                initRos = True
        else:
            print 'Waiting for the ros master'
            if initRos:
                print 'Shudowning the ros node'
                s.shutdown();
                #rospy.signal_shutdown('Ros master is shutdown')
                initRos = False
            else:
                time.sleep(1)
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, 
        help='Could be either a directory containing the model_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_file', type=str,
        help='Classifier model file name as a pickle (.pkl) file.)')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--threshold_reco', type=int,
        help='Threshold to face recognition.', default=0.9)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

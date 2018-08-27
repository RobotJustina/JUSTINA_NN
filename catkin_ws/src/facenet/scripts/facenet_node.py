#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from scipy import misc
import os
import re
import argparse
import tensorflow as tf
import align.detect_face
import numpy as np
import math
import pickle
import facenet

import numpy as np
import face_recognition as face
import tensorflow as tf

def facenet_align(cv_image):
    #print("facnet align function")
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7] # three step's threshold
    #threshold = [0.65, 0.75, 0.75] # three step's threshold
    factor = 0.709 # scale factor

    bounding_boxes, _ = align.detect_face.detect_face(cv_image, minsize, pnet, rnet, onet, threshold, factor)
    
    nrof_faces = bounding_boxes.shape[0]
    
    det_arr = []
    if nrof_faces > 0:
        #print ("Faces detected:" + str(nrof_faces))
        #print (bounding_boxes)
        det = bounding_boxes[:,0:4]
        #print (det)
        img_size = np.asarray(cv_image.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0] + det[:,2])/2 - img_center[1], (det[:,1] + det[:,3])/2 - img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))
    return det_arr

def facenet_recognition(det, cv_image):
    with facenetGraph.as_default(): 
        with facenetSession.as_default() as sess:
            img_size = np.asarray(cv_image.shape)[0:2]
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin/2, 0)
            bb[1] = np.maximum(det[1] - margin/2, 0)
            bb[2] = np.minimum(det[2] + margin/2, img_size[1])
            bb[3] = np.minimum(det[3] + margin/2, img_size[0])
            cropped = cv_image[bb[1]:bb[3], bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            cv2.imshow("Image", cv_image)
            image = facenet.prewhiten(scaled)
            feed_dict = { images_placeholder:[image], phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            
            predictions = model.predict_proba([emb_array[0]])
            print (predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            #print (best_class_indices)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
               
    return (class_names[best_class_indices[0]], best_class_probabilities[0])

def compressed_image_callback(img):
    try:
        img_np_arr = np.fromstring(img.data, np.uint8)
        encoded_img = cv2.imdecode(img_np_arr, 1)
        flipped_img = cv2.flip(encoded_img, 1)
        cv_image = flipped_img
    except CvBridgeError as e:
        print(e)

    with facenetGraph.as_default():
        with face_recognition.encoder.sess.as_default():
            faces = face_recognition.identify(cv_image)

    add_overlays(cv_image, faces)
       
    cv2.imshow("Image", cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
    
def image_callback(img):
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    det_arr = facenet_align(cv_image)
    img_size = np.asarray(cv_image.shape)[0:2]

    classes = []
    probabilities = []

    if len(det_arr) > 0:
        for i, det in enumerate(det_arr):
            (class_name, probability) = facenet_recognition(det, cv_image)
            classes.append(class_name)
            probabilities.append(probability)
    
    if len(det_arr) > 0:
        for i, det in enumerate(det_arr):
            (class_name, probability) = facenet_recognition(det, cv_image)
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin/2, 0)
            bb[1] = np.maximum(det[1] - margin/2, 0)
            bb[2] = np.minimum(det[2] + margin/2, img_size[1])
            bb[3] = np.minimum(det[3] + margin/2, img_size[0])
            cv2.rectangle(cv_image,(bb[0], bb[1]),(bb[2],bb[3]),(0,255,0),3)
            if probabilities[i] > threshold_reco:
                cv2.putText(cv_image, classes[i].split(' ')[0], (bb[0], bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,255),1,cv2.LINE_AA)
            else:
                cv2.putText(cv_image, 'Unknown', (bb[0], bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1,cv2.LINE_AA)
  
    cv2.imshow("Image", cv_image)
    cv2.waitKey(1)

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


def main(args):
    rospy.init_node('facenet_node', anonymous=True)
    rate = rospy.Rate(30) #30Hz

    ##global pnet
    ##global rnet
    ##global onet

    global bridge
    global facenetGraph
    global face_recognition

    global threshold_reco
    
    ##global margin
    ##global image_size
    ##global detect_multiple_faces

    ##global facenetGraph
    ##global facenetSession

    ##global model
    ##global class_names
   
    model_file = rospy.get_param("~model_file");
    classifier_file = rospy.get_param("~classifier_file")
    image_size = rospy.get_param("~image_size")
    margin = rospy.get_param("~margin")
    gpu_memory_fraction = rospy.get_param("~gpu_memory_fraction")
    detect_multiple_faces = rospy.get_param("~detect_multiple_faces")
    threshold_reco = rospy.get_param("~threshold_reco")

    ##with tf.Graph().as_default():
    ##    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    ##    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    ##    with sess.as_default():
    ##        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    ##facenetGraph = tf.Graph()
    ##with facenetGraph.as_default():
    ##    facenetSession = tf.Session()
    ##    with facenetSession.as_default() as sess:
    ##        facenet.load_model(model_file)
    ##        classifier_filename_exp = os.path.expanduser(classifier_file)
    ##        with open(classifier_filename_exp, 'rb') as infile:
    ##            (model, class_names) = pickle.load(infile)
    ##            print('Loaded classifier model from file "%s"' % classifier_filename_exp);
    
    bridge = CvBridge()

    facenetGraph = tf.Graph()
    with facenetGraph.as_default():
        face_recognition = face.Recognition(facenet_model=model_file, classifier_model=classifier_file, face_crop_size=image_size, threshold=[0.7,0.8,0.8], factor=0.709, face_crop_margin=margin, gpu_memory_fraction=gpu_memory_fraction, detect_multiple_faces=detect_multiple_faces)
 
    ##rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
    ##rospy.spin()

    #rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, compressed_image_callback, queue_size=1)
                                                                                                                                 
    while not rospy.is_shutdown():
        print("Waiting for image\n")
        img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage)
        img_np_arr = np.fromstring(img.data, np.uint8)
        encoded_img = cv2.imdecode(img_np_arr, 1)
        flipped_img = cv2.flip(encoded_img, 1)
        with facenetGraph.as_default():
            with face_recognition.encoder.sess.as_default():
                faces = face_recognition.identify(flipped_img)
        add_overlays(flipped_img, faces)
       
        cv2.imshow('Face Recognition', flipped_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv[1:])
    

#!/usr/bin/env python
from std_msgs.msg import Header
from std_msgs.msg import String
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from justina_nn_msgs.msg import *
from justina_nn_msgs.srv import *
from rospkg import RosPack

import sys
import rospy
import rosgraph
import os
import cv2
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED) 
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def load_network(model_path, image_size, gpu_memory_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    images_pl = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8, phase_train=train_mode,weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
    tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess,age,gender,train_mode,images_pl

def faces_age_gender_callback(req):
    head = std_msgs.msg.Header()
    head.stamp = rospy.Time.now()
    try:
        cv_image = bridge.imgmsg_to_cv2(req.imageBGR, "bgr8")
        input_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)
    except CvBridgeError as e:
        print(e)

    if len(req.faces.recog_faces) == 0:
    
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), image_size, image_size, 3))

        recog_faces = [] 
        img_size = np.asarray(cv_image.shape)[0:2]
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
            # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            bounding_box = [Point(x1, y1, 0), Point(x2, y2, 0)]
            face_centroid = Point((x1 + x2) / 2, (y1 + y2) / 2, 0)
            face_class = VisionFaceObject(face_centroid=face_centroid, bounding_box=bounding_box)
            recog_faces.append(face_class);
    
    else:
        faces = np.empty((len(req.faces.recog_faces), image_size, image_size, 3))
        img_size = np.asarray(cv_image.shape)[0:2]

        for i, fr in enumerate(req.faces.recog_faces):
            d = dlib.rectangle(left=int(fr.bounding_box[0].x), top=int(fr.bounding_box[0].y), right=int(fr.bounding_box[1].x), bottom=int(fr.bounding_box[1].y))
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            faces[i, :, :, :] = fa.align(input_img, gray, d)
        
        recog_faces = req.faces.recog_faces
    
    if len(recog_faces) > 0:
        # predict ages and genders of the detected faces
        ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
    
    # draw results
    for i, d in enumerate(recog_faces):
        label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
        recog_faces[i].ages = int(ages[i])
        recog_faces[i].gender = int(genders[i])
        #draw_label(encoded_img, (d.left(), d.top()), label)

    #cv2.imshow('Face Recognition', cv_image)
    #cv2.waitKey(1)

    return FaceRecognitionResponse(VisionFaceObjects(head, recog_faces))

def main(args):
    rospy.init_node('face_age_gender_node', anonymous=True)
    rate = rospy.Rate(30) #30Hz

    global detector
    global predictor
    global fa

    global bridge
    global sess
    global age
    global gender
    global train_mode
    global images_pl
    global image_size

    model_path = rospy.get_param("~model_path")
    face_landmarks = rospy.get_param("~face_landmarks_file")
    weight_file = rospy.get_param("~weight_file")
    depth = rospy.get_param("~depth")
    width = rospy.get_param("~width")
    image_size = rospy.get_param("~image_size")
    gpu_memory_fraction = rospy.get_param("~gpu_memory_fraction")
    
    bridge = CvBridge()

    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmarks)
    fa = FaceAligner(predictor, desiredFaceWidth=image_size)
    
    sess, age, gender, train_mode,images_pl = load_network(model_path, image_size, gpu_memory_fraction)

    s = rospy.Service('face_recognizer/faces_age_gender', FaceRecognition, faces_age_gender_callback)
 
    while not rospy.is_shutdown() and rosgraph.is_master_online():
        print("Waiting for image\n")
        try:
            img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage, timeout=1)
            img_np_arr = np.fromstring(img.data, np.uint8)
            encoded_img = cv2.imdecode(img_np_arr, 1)
            input_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2GRAY)
            img_h, img_w, _ = np.shape(input_img)
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), image_size, image_size, 3))
            
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(encoded_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
                # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        
            if len(detected) > 0:
                # predict ages and genders of the detected faces
                ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
                draw_label(encoded_img, (d.left(), d.top()), label)

            cv2.imshow('Face Recognition', encoded_img)
    	    if cv2.waitKey(1) & 0xFF == ord('q'):
            	return
        
        except rospy.ROSException as e:
            pass

        rate.sleep()

if __name__ == '__main__':
    main(sys.argv[1:])

#!/usr/bin/env python

import sys
import rospy
import rosgraph
import cv2
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

from scipy import misc
import os
import re
import glob
import shutil
import argparse
import tensorflow as tf
import align.detect_face
import numpy as np
import math
import pickle
import facenet

import numpy as np
#import face_recognition_classifier as face
import face_recognition as face
import face_recognition_classifier as faceC
import tensorflow as tf
from sklearn.svm import LinearSVC

def sorted_nicely(strings):
    return sorted(strings, key=natural_sort_key)

def natural_sort_key(key):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 0, 0), 2)
            print ("---Face name ----" + str(face.name))
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
                #cv_image = cv2.flip(cv_image, 1)
            except CvBridgeError as e:
                print(e)
            faces = face_recognition.identify(cv_image)
    #add_overlays(cv_image, faces)
    
    recog_faces = [] 
    for face in faces:
        if not face_recog_available:
            face.name = 'Unknown'
            face.probability = 1.0
        if(req.id == '' or face.name == req.id.replace("_"," ")):
            face_name = face.name
            confidence = face.probability
            if(face.probability <= threshold_reco and req.id == ''):
                face_name = "Unknown"
            elif(req.id != '' and face.probability <= threshold_reco):
                continue
                
            img_size = np.asarray(cv_image.shape)[0:2]
            #bounding_box = [Point(img_size[1] - face.bounding_box[2], face.bounding_box[1], 0), Point(img_size[1] - face.bounding_box[0], face.bounding_box[3], 0)]
            #face_centroid = Point(img_size[1] - (face.bounding_box[0] + face.bounding_box[2]) / 2, (face.bounding_box[1] + face.bounding_box[3]) / 2, 0)
            bounding_box = [Point(face.bounding_box[0], face.bounding_box[1], 0), Point(face.bounding_box[2], face.bounding_box[3], 0)]
            face_centroid = Point((face.bounding_box[0] + face.bounding_box[2]) / 2, (face.bounding_box[1] + face.bounding_box[3]) / 2, 0)
            face_class = VisionFaceObject(id=face_name, confidence=confidence, face_centroid=face_centroid, bounding_box=bounding_box)
            recog_faces.append(face_class);

    #cv2.imshow('Face Recognition', cv_image)
    #cv2.waitKey(1)

    return FaceRecognitionResponse(VisionFaceObjects(h, recog_faces))

def add_face_callback(req):
    try:
        cv_image = bridge.imgmsg_to_cv2(req.imageBGR, "bgr8")
    except CvBridgeError as e:
        print(e)
    add_face_to_train(cv_image, req.id)
    return FaceRecognitionResponse()

def train_faces_callback(req):
    train_faces_classifier()
    return EmptySrvResponse()

def add_face_to_train(image, name):
    print ('Trainig person' + name)
    #image = cv2.flip(image, 1)
    old_detect_multiple_faces = face_recognition.detect.detect_multiple_faces
    face_recognition.detect.detect_multiple_faces = False

    person_dir = training_dir + "/" + name
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    reg_exp = person_dir + "/" + name + "_[0-9]*.png"
    result = sorted_nicely( glob.glob(reg_exp))
    with facenetGraph.as_default():
        with face_recognition.detect.sess.as_default():
            faces = face_recognition.detect.find_faces(image)
    for face in faces:
        if (len(result) == 0):
            filename = person_dir + "/" + name + "_0.png";
        else:
            last_result = result[-1]
            number = re.search( person_dir + "/" + name + "_([0-9]*).png",last_result).group(1)
            filename = person_dir + "/" + name + "_%i.png"%+(int(number)+1)
        misc.imsave(filename, cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB))
        if not classifier_mode: 
            with facenetGraph.as_default():
                with face_recognition.encoder.sess.as_default():
                    face.embedding = face_recognition.encoder.generate_embedding(face)
                    face_recognition.identifier.db_emb = np.concatenate((face_recognition.identifier.db_emb, [face.embedding]))
                    try:
                        index = face_recognition.identifier.class_names.index(name)
                    except ValueError:
                        print("Add the new class name")
                        face_recognition.identifier.class_names.append(name)
                        if len(face_recognition.identifier.labels) > 0:
                            index = face_recognition.identifier.labels[np.argmax(face_recognition.identifier.labels)] + 1
                        else:
                            index = 0
                    face_recognition.identifier.labels.append(index)
    face_recognition.detect.detect_multiple_faces = old_detect_multiple_faces

def train_faces_classifier():
    global face_recog_available
    dataset = facenet.get_dataset(training_dir)
    for cls in dataset:
        assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
    paths, labels = facenet.get_image_paths_and_labels(dataset)

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))
    
    if len(paths) > 1:
        with facenetGraph.as_default():
            with face_recognition.detect.sess.as_default():
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
            with face_recognition.encoder.sess.as_default():
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min( (i+1) * batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = face_recognition.encoder.sess.run(embeddings, feed_dict=feed_dict)
                face_recognition.identifier.mode = LinearSVC()
                face_recognition.identifier.model.fit(emb_array, labels)
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                face_recognition.identifier.class_names = class_names
                face_recog_available = True
    else:
        face_recog_available = False

def clear_faces_callback(msg):
    global clear_bd
    clear_bd = True

def clear_face_id_callback(msg):
    global clear_face_id
    global face_clear_id
    clear_face_id = True
    face_clear_id = msg.data

def main(args):
    rospy.init_node('facenet_node', anonymous=True)
    rospack = RosPack()
    rate = rospy.Rate(30) #30Hz

    global bridge
    global facenetGraph
    global face_recognition

    global classifier_file
    global training_dir
    
    global threshold_reco

    global image_size
    global training_dir
    global batch_size
    global face_recog_available
    global classifier_mode

    global clear_face_id
    global clear_bd
    global face_clear_id

    clear_face_id = False
    clear_bd = False
    face_clear_id = ""
    face_recog_available = True
    
    model_file = rospy.get_param("~model_file");
    classifier_file = rospy.get_param("~classifier_file")
    image_size = rospy.get_param("~image_size")
    margin = rospy.get_param("~margin")
    gpu_memory_fraction = rospy.get_param("~gpu_memory_fraction")
    detect_multiple_faces = rospy.get_param("~detect_multiple_faces")
    threshold_reco = rospy.get_param("~threshold_reco")
    training_dir = rospy.get_param("~training_dir")
    classifier_mode = rospy.get_param("~classifier_mode")
    batch_size = rospy.get_param("~batch_size")
    
    try:  
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % training_dir)
    else:  
        print ("Successfully created the directory %s " % training_dir)

    if classifier_mode:
        try:
            if not os.path.exists(training_dir + "/Unknown/defualt.png"):
                package_path = rospack.get_path('facenet_ros')
                image_path = package_path + "/default.png"
                img_def = cv2.imread(image_path, 1)
                if not os.path.exists(training_dir + "/Unknown"):
                    os.makedirs(training_dir + "/Unknown")
                misc.imsave(training_dir + "/Unknown/default.png", cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB))
        except OSError:  
            print ("Creation of the directory %s failed" % training_dir)
        else:  
            print ("Successfully created the directory %s " % training_dir)
    else:
        try:
            if os.path.exists(training_dir + "/Unknown/"):
                shutil.rmtree(training_dir + "/Unknown/")
        except OSError:
            pass

    bridge = CvBridge()

    facenetGraph = tf.Graph()
    with facenetGraph.as_default():
        if classifier_mode:
            face_recognition = faceC.Recognition(facenet_model=model_file, classifier_model=classifier_file, face_crop_size=image_size, threshold=[0.7,0.8,0.8], factor=0.709, face_crop_margin=margin, gpu_memory_fraction=gpu_memory_fraction, detect_multiple_faces=detect_multiple_faces)
        else:
            face_recognition = face.Recognition(facenet_model=model_file, data_dir=training_dir, face_crop_size=image_size, threshold=[0.7,0.8,0.8], factor=0.709, face_crop_margin=margin, gpu_memory_fraction=gpu_memory_fraction, detect_multiple_faces=detect_multiple_faces, batch_size=batch_size)

    s = rospy.Service('face_recognizer/faces', FaceRecognition, face_recognition_callback)
    sa = rospy.Service('face_recognizer/train_face', FaceRecognition, add_face_callback)
    if classifier_mode:
        st = rospy.Service('face_recognizer/train_flush', EmptySrv, train_faces_callback)
    rospy.Subscriber("face_recognizer/clear_faces", Empty, clear_faces_callback)
    rospy.Subscriber("face_recognizer/clear_face_id", String, clear_face_id_callback)
 
    while not rospy.is_shutdown() and rosgraph.is_master_online():
        print("Waiting for image\n")
        try:
            img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage, timeout=1)
            img_np_arr = np.fromstring(img.data, np.uint8)
            encoded_img = cv2.imdecode(img_np_arr, 1)
            #encoded_img = cv2.flip(encoded_img, 1)
            with facenetGraph.as_default():
                with face_recognition.encoder.sess.as_default():
                    faces = face_recognition.identify(encoded_img)
            add_overlays(encoded_img, faces)
       
            cv2.imshow('Face Recognition', encoded_img)
    	    if cv2.waitKey(1) & 0xFF == ord('q'):
            	return
        
        except rospy.ROSException as e:
            pass
        if clear_bd or clear_face_id:
            print ('Try to clean a data base')
            if os.path.exists(training_dir):
                if clear_bd:
                    for the_file in os.listdir(training_dir):
                        file_path = os.path.join(training_dir, the_file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path) and (not "Unknown" in os.path.basename(file_path) or not classifier_mode) : shutil.rmtree(file_path)
                        except Exception as e:
                            print(e)
                    if classifier_mode:
                        train_faces_classifier()
                    else:
                        face_recognition.identifier.class_names = []
                        face_recognition.identifier.labels = []
                        if len(face_recognition.identifier.db_emb) > 0:
                            face_recognition.identifier.db_emb = np.empty((0, len(face_recognition.identifier.db_emb[0])))
                if clear_face_id:
                    person_dir = training_dir + "/" + face_clear_id
                    if os.path.exists(person_dir):
                        shutil.rmtree(person_dir)
                        if classifier_mode:
                            train_faces_classifier()
                    if not classifier_mode:
                        try:
                            index_class = face_recognition.identifier.class_names.index(face_clear_id)
                            face_recognition.identifier.class_names.remove(face_clear_id)
                            indices = [i for i, x in enumerate(face_recognition.identifier.labels) if x == index_class]
                            indices.sort()
                            removes = 0
                            if len(indices) > 0:
                                for index in range(len(indices)):
                                    del face_recognition.identifier.labels[indices[index] + removes]
                                    removes -= 1
                                face_recognition.identifier.db_emb = np.delete(face_recognition.identifier.db_emb, indices , 0)
                        except ValueError:
                            pass
            clear_bd = False
            clear_face_id = False
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv[1:])
    

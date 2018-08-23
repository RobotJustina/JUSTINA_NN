#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
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

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def facenet_align(cv_image):
    #print("facnet align function")
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7] # three step's threshold
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

def facenet_recognition(det_arr, cv_image):
    with facenetGraph.as_default(): 
        with facenetSession.as_default() as sess:
            img_size = np.asarray(cv_image.shape)[0:2]
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            #print('Calculating features for images')
            nrof_images = len(det_arr)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i+1) * batch_size, nrof_images)
                det_arr_batch = det_arr[start_index:end_index]
                #images = facenet.load_data(paths_batch, False, False, args.image_size)
                images = []
                for i, det in enumerate(det_arr_batch):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin/2, 0)
                    bb[1] = np.maximum(det[1] - margin/2, 0)
                    bb[2] = np.minimum(det[2] + margin/2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin/2, img_size[0])
                    cropped = cv_image[bb[1]:bb[3], bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    images.append(scaled)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            predictions = model.predict_proba(emb_array)
            #print (predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            print (best_class_indices)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
    
def image_callback(data):
        try:
            cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        det_arr = facenet_align(cv_image)
        img_size = np.asarray(cv_image.shape)[0:2]

        nrof_successfully_aligned = 0

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin/2, 0)
            bb[1] = np.maximum(det[1] - margin/2, 0)
            bb[2] = np.minimum(det[2] + margin/2, img_size[1])
            bb[3] = np.minimum(det[3] + margin/2, img_size[0])
            #cropped = cv_image[bb[1]:bb[3], bb[0]:bb[2],:]
            #scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            cv2.rectangle(cv_image,(bb[0], bb[1]),(bb[2],bb[3]),(0,255,0),3)
            nrof_successfully_aligned += 1
            #filename_base, file_extension = os.path.splitext(output_filename)
            #if args.detect_multiple_faces:
            #    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
            #else:
            #output_filename_n = "{}{}".format(filename_base, file_extension)
            #misc.imsave(output_filename_n, scaled)
            #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))

        if nrof_successfully_aligned > 0:
            facenet_recognition(det_arr, cv_image)
        
        cv2.imshow("Image", cv_image)
        cv2.waitKey(3)

def main(args):
    rospy.init_node('facenet_node', anonymous=True)
    rate = rospy.Rate(3) #30Hz

    global bridge
    
    global pnet
    global rnet
    global onet

    global margin
    global image_size
    global batch_size
    global detect_multiple_faces

    global facenetGraph
    global facenetSession

    global model
    global class_names

    model_file = rospy.get_param("~model_file");
    classifier_file = rospy.get_param("~classifier_file")
    image_size = rospy.get_param("~image_size")
    margin = rospy.get_param("~margin")
    gpu_memory_fraction = rospy.get_param("~gpu_memory_fraction")
    detect_multiple_faces = rospy.get_param("~detect_multiple_faces")
    batch_size = rospy.get_param("~batch_size")

    bridge = CvBridge()
 
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    facenetGraph = tf.Graph()
    with facenetGraph.as_default():
        facenetSession = tf.Session()
        with facenetSession.as_default() as sess:
            load_model(model_file)
            classifier_filename_exp = os.path.expanduser(classifier_file)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

    rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
                                                                                                                                 
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv[1:])
    

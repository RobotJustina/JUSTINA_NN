#!/usr/bin/env python

# py2/3 compability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

import rospy
import rosgraph
from rospkg import RosPack

import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from mmrcnn.config import Config
from mmrcnn import model as modellib, utils
from mmrcnn import coco
from mmrcnn import visualize
    
class LikeCocoConfig(Config):

    """Configuration for training on the coco  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "LikeCoco"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + balloon

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'mobilenetv1'
    #BACKBONE = 'resnet50'

    USE_MULTIPROCESSING = True

def main(args):
    rospy.init_node("mask_rcnn_node", anonymous=True)
    rospack = RosPack()
    rate = rospy.Rate(30)
    
    weights_path = rospy.get_param("~weights_path")
    logs_path = rospy.get_param("~logs_path")

    nameConfig = "coco"
     
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light',
                                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                                                         'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                                                                        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                                                                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                                                                                      'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                                                                                                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                                                                                                                                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                                                                                                                                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                                                                                                                                                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                                                                                                                                                                                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                                                                                                                                                                                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                                                                                                                                                                                                               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                                                                                                                                                                                                              'teddy bear', 'hair drier', 'toothbrush']


    config = coco.CocoConfig();
    #config = LikeCocoConfig()

    class InferenceConfig(config.__class__):
        NAME = nameConfig
        #BACKBONE = "resnet50"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        #IMAGE_MIN_DIM = 512
        #IMAGE_MAX_DIM = 512
        DETECTION_MIN_CONFIDENCE = 0.85
 
    config = InferenceConfig()
    config.display()
    
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs_path)
    model.load_weights(weights_path, by_name=True)

    while not rospy.is_shutdown() and rosgraph.is_master_online():
        print("Waiting for image\n")
        try:
            img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage, timeout=1)
            img_np_arr = np.fromstring(img.data, np.uint8)
            encoded_img = cv2.imdecode(img_np_arr, 1)
            #encoded_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB) 
            results = model.detect([encoded_img], verbose=1)

            r = results[0]
            print(len(r))
            vis_image = visualize.draw_instances(encoded_img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            plt.show()
            cv2.imshow('Mask RCNN', vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        except rospy.ROSException as e:
            pass
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv[1:])

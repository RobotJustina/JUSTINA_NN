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

	"""Base configuration class. For custom configurations, create a
    	sub-class that inherits from this one and override properties
    	that need to be changed.
    	"""
    	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    	# Useful if your code needs to do things differently depending on which
    	# experiment is running.

    	# Number of images to train with on each GPU. A 12GB GPU can typically
    	# handle 2 images of 1024x1024px.
    	# Adjust based on your GPU memory and image sizes. Use the highest
    	# number that your GPU can handle for best performance.
    	IMAGES_PER_GPU = 1
	GPU_COUNT = 1

    	# Number of training steps per epoch
    	# This doesn't need to match the size of the training set. Tensorboard
    	# updates are saved at the end of each epoch, so setting this to a
    	# smaller number means getting more frequent TensorBoard updates.
    	# Validation stats are also calculated at each epoch end and they
    	# might take a while, so don't set this too small to avoid spending
    	# a lot of time on validation stats.
    	STEPS_PER_EPOCH = 1000

    	# Number of validation steps to run at the end of every training epoch.
    	# A bigger number improves accuracy of validation stats, but slows
    	# down the training.
    	VALIDATION_STEPS = 50

    	# Backbone network architecture
    	# Supported values are: resnet50, resnet101.
    	# You can also provide a callable that should have the signature
    	# of model.resnet_graph. If you do so, you need to supply a callable
    	# to COMPUTE_BACKBONE_SHAPE as well
    	BACKBONE = "resnet101"

    	# Only useful if you supply a callable to BACKBONE. Should compute
    	# the shape of each layer of the FPN Pyramid.
    	# See model.compute_backbone_shapes
    	COMPUTE_BACKBONE_SHAPE = None

    	# The strides of each layer of the FPN Pyramid. These values
    	# are based on a Resnet101 backbone.
    	BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    	# Size of the fully-connected layers in the classification graph
    	FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    	# Size of the top-down layers used to build the feature pyramid
    	TOP_DOWN_PYRAMID_SIZE = 256


    	# Length of square anchor side in pixels
    	RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    	# Ratios of anchors at each cell (width/height)
    	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
    	RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    	# Anchor stride
    	# If 1 then anchors are created for each cell in the backbone feature map.
    	# If 2, then anchors are created for every other cell, and so on.
    	RPN_ANCHOR_STRIDE = 1

    	# Non-max suppression threshold to filter RPN proposals.
    	# You can increase this during training to generate more propsals.
    	RPN_NMS_THRESHOLD = 0.7

    	# How many anchors per image to use for RPN training
    	RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    	# ROIs kept after tf.nn.top_k and before non-maximum suppression
    	PRE_NMS_LIMIT = 6000

    	# ROIs kept after non-maximum suppression (training and inference)
    	POST_NMS_ROIS_TRAINING = 2000
    	POST_NMS_ROIS_INFERENCE = 1000

    	# If enabled, resizes instance masks to a smaller size to reduce
    	# memory load. Recommended when using high-resolution images.
    	USE_MINI_MASK = True
    	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    	# Input image resizing
    	# Generally, use the "square" resizing mode for training and predicting
    	# and it should work well in most cases. In this mode, images are scaled
    	# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    	# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    	# padded with zeros to make it a square so multiple images can be put
    	# in one batch.
    	# Available resizing modes:
    	# none:   No resizing or padding. Return the image unchanged.
    	# square: Resize and pad with zeros to get a square image
    	#         of size [max_dim, max_dim].
    	# pad64:  Pads width and height with zeros to make them multiples of 64.
    	#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    	#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    	#         The multiple of 64 is needed to ensure smooth scaling of feature
    	#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    	# crop:   Picks random crops from the image. First, scales the image based
    	#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    	#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    	#         IMAGE_MAX_DIM is not used in this mode.
    	IMAGE_RESIZE_MODE = "square"
    	IMAGE_MIN_DIM = 800
    	IMAGE_MAX_DIM = 1024
    	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    	# up scaling. For example, if set to 2 then images are scaled up to double
    	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    	# However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    	IMAGE_MIN_SCALE = 0
    	# Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    	# Changing this requires other changes in the code. See the WIKI for more
    	# details: https://github.com/matterport/Mask_RCNN/wiki
    	IMAGE_CHANNEL_COUNT = 3

    	# Image mean (RGB)
    	MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    	# Number of ROIs per image to feed to classifier/mask heads
    	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
    	# enough positive proposals to fill this and keep a positive:negative
    	# ratio of 1:3. You can increase the number of proposals by adjusting
    	# the RPN NMS threshold.
    	TRAIN_ROIS_PER_IMAGE = 200

    	# Percent of positive ROIs used to train classifier/mask heads
    	ROI_POSITIVE_RATIO = 0.33

    	# Pooled ROIs
    	POOL_SIZE = 7
    	MASK_POOL_SIZE = 14

    	# Shape of output mask
    	# To change this you also need to change the neural network mask branch
    	MASK_SHAPE = [28, 28]

    	# Maximum number of ground truth instances to use in one image
    	MAX_GT_INSTANCES = 100

    	# Bounding box refinement standard deviation for RPN and final detections.
    	RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    	BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    	# Max number of final detections
    	DETECTION_MAX_INSTANCES = 100

    	# Minimum probability value to accept a detected instance
    	# ROIs below this threshold are skipped
    	DETECTION_MIN_CONFIDENCE = 0.7

    	# Non-maximum suppression threshold for detection
    	DETECTION_NMS_THRESHOLD = 0.3

    	# Learning rate and momentum
    	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    	# weights to explode. Likely due to differences in optimizer
    	# implementation.
    	LEARNING_RATE = 0.001
    	LEARNING_MOMENTUM = 0.9

    	# Weight decay regularization
    	WEIGHT_DECAY = 0.0001

    	# Loss weights for more precise optimization.
    	# Can be used for R-CNN training setup.
    	LOSS_WEIGHTS = {
        	"rpn_class_loss": 1.,
        	"rpn_bbox_loss": 1.,
        	"mrcnn_class_loss": 1.,
        	"mrcnn_bbox_loss": 1.,
        	"mrcnn_mask_loss": 1.
    	}

    	# Use RPN ROIs or externally generated ROIs for training
    	# Keep this True for most situations. Set to False if you want to train
    	# the head branches on ROI generated by code rather than the ROIs from
    	# the RPN. For example, to debug the classifier head without having to
    	# train the RPN.
    	USE_RPN_ROIS = True

    	# Train or freeze batch normalization layers
    	#     None: Train BN layers. This is the normal mode
    	#     False: Freeze BN layers. Good when using a small batch size
    	#     True: (don't use). Set layer in training mode even when predicting
    	TRAIN_BN = False  # Defaulting to False since batch size is often small

    	# Gradient norm clipping
    	GRADIENT_CLIP_NORM = 5.0
        DETECTION_MIN_CONFIDENCE = 0.85
	BATCH_SIZE = 1
 
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

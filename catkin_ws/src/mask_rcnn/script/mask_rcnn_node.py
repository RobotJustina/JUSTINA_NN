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
    NAME = "LikeCoco"

def main(args):
    rospy.init_node("mask_rcnn_node", anonymous=True)
    rospack = RosPack()
    rate = rospy.Rate(30)
   
    class_names = rospy.get_param("/mask_rcnn/mask_rcnn_model/class_names")
    weights_file = rospy.get_param("~weights_path") + rospy.get_param("/mask_rcnn/mask_rcnn_model/weights_name")
    logs_path = rospy.get_param("~logs_path") + rospy.get_param("/mask_rcnn/mask_rcnn_model/logs_name")
    name_model = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/NAME")

    #class_names = ['BG', 'blue_knife', 'yellow_knife', 'orange_knife', 'green_knife', 'red_knife', 'purple_knife', 'blue_fork', 'yellow_fork', 'orange_fork', 'green_fork', 'red_fork', 'purple_fork', 'blue_spoon', 'yellow_spoon', 'orange_spoon', 'green_spoon', 'red_spoon', 'purple_spoon', 'blue_dish', 'yellow_dish', 'orange_dish', 'green_dish', 'red_dish', 'purple_dish', 'blue_bowl', 'yellow_bowl', 'orange_bowl', 'green_bowl', 'red_bowl', 'purple_bowl', 'blue_glass', 'yellow_glass', 'orange_glass', 'green_glass', 'red_glass', 'purple_glass']
    # class_names = ['BG', 'cereal', 'papikra', 'pringles' , 'potato_chips', 'chocolate_drink', 'coke', 'grape_juice', 'orange_juice', 'tray']

    if name_model == 'coco':
        config = coco.CocoConfig()
    else:
        config = LikeCocoConfig()

    class InferenceConfig(config.__class__):

    	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    	# Useful if your code needs to do things differently depending on which
    	# experiment is running.
    	NAME = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/NAME")  # Override in sub-classes

    	# NUMBER OF GPUs to use. For CPU training, use 1
    	GPU_COUNT = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/GPU_COUNT")

    	# Use Multiprocessing in MaskRCNN.train()
    	USE_MULTIPROCESSING = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/USE_MULTIPROCESSING")

    	# Number of images to train with on each GPU. A 12GB GPU can typically
    	# handle 2 images of 1024x1024px.
    	# Adjust based on your GPU memory and image sizes. Use the highest
    	# number that your GPU can handle for best performance.
    	IMAGES_PER_GPU = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/IMAGES_PER_GPU")

    	# Number of training steps per epoch
    	# This doesn't need to match the size of the training set. Tensorboard
    	# updates are saved at the end of each epoch, so setting this to a
    	# smaller number means getting more frequent TensorBoard updates.
    	# Validation stats are also calculated at each epoch end and they
    	# might take a while, so don't set this too small to avoid spending
    	# a lot of time on validation stats.
    	STEPS_PER_EPOCH = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/STEPS_PER_EPOCH")

    	# Number of validation steps to run at the end of every training epoch.
    	# A bigger number improves accuracy of validation stats, but slows
    	# down the training.
    	VALIDATION_STEPS = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/VALIDATION_STEPS")

    	# Backbone Architecture,
    	# Currently supported: ['resnet50','resnet101', 'mobilenetv1','mobilenetv2']
    	BACKBONE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/BACKBONE") # Override in sub-classes

	# Only useful if you supply a callable to BACKBONE. Should compute
    	# the shape of each layer of the FPN Pyramid.
    	# See model.compute_backbone_shapes
    	COMPUTE_BACKBONE_SHAPE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/COMPUTE_BACKBONE_SHAPE")

    	# The strides of each layer of the FPN Pyramid. These values
    	# are based on a Resnet101 backbone.
    	BACKBONE_STRIDES = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/BACKBONE_STRIDES") #resnet

    	# Number of classification classes (including background)
    	NUM_CLASSES = 1 + rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/NUM_CLASSES") # Override in sub-classes

	# Size of the fully-connected layers in the classification graph
        FPN_CLASSIF_FC_LAYERS_SIZE =  rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/FPN_CLASSIF_FC_LAYERS_SIZE")

	# Size of the top-down layers used to build the feature pyramid
    	TOP_DOWN_PYRAMID_SIZE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/TOP_DOWN_PYRAMID_SIZE")

    	# Length of square anchor side in pixels
    	RPN_ANCHOR_SCALES = (rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_SCALES")[0], rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_SCALES")[1], rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_SCALES")[2], rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_SCALES")[3], rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_SCALES")[4])

    	# Ratios of anchors at each cell (width/height)
    	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
    	RPN_ANCHOR_RATIOS = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_RATIOS")

    	# Anchor stride
    	# If 1 then anchors are created for each cell in the backbone feature map.
    	# If 2, then anchors are created for every other cell, and so on.
    	RPN_ANCHOR_STRIDE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_ANCHOR_STRIDE")

    	# Non-max suppression threshold to filter RPN proposals.
    	# You can increase this during training to generate more propsals.
    	RPN_NMS_THRESHOLD = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_NMS_THRESHOLD")

    	# How many anchors per image to use for RPN training
    	RPN_TRAIN_ANCHORS_PER_IMAGE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_TRAIN_ANCHORS_PER_IMAGE")

	# ROIs kept after tf.nn.top_k and before non-maximum suppression
	PRE_NMS_LIMIT = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/PRE_NMS_LIMIT")

    	# ROIs kept after non-maximum supression (training and inference)
    	POST_NMS_ROIS_TRAINING = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/POST_NMS_ROIS_TRAINING")
    	POST_NMS_ROIS_INFERENCE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/POST_NMS_ROIS_INFERENCE")

    	# If enabled, resizes instance masks to a smaller size to reduce
    	# memory load. Recommended when using high-resolution images.
    	USE_MINI_MASK = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/USE_MINI_MASK")
    	MINI_MASK_SHAPE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/MINI_MASK_SHAPE")  # (height, width) of the mini-mask

    	# Input image resizing
    	# Generally, use the "square" resizing mode for training and inferencing
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
    	IMAGE_RESIZE_MODE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/IMAGE_RESIZE_MODE")
    	IMAGE_MIN_DIM = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/IMAGE_MIN_DIM")
    	IMAGE_MAX_DIM = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/IMAGE_MAX_DIM")
    	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    	# up scaling. For example, if set to 2 then images are scaled up to double
    	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    	# Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    	IMAGE_MIN_SCALE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/IMAGE_MIN_SCALE")

    	# Image mean (RGB)
    	MEAN_PIXEL = np.array(rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/MEAN_PIXEL"))

    	# Number of ROIs per image to feed to classifier/mask heads
    	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
    	# enough positive proposals to fill this and keep a positive:negative
    	# ratio of 1:3. You can increase the number of proposals by adjusting
    	# the RPN NMS threshold.
    	TRAIN_ROIS_PER_IMAGE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/TRAIN_ROIS_PER_IMAGE")

    	# Percent of positive ROIs used to train classifier/mask heads
    	ROI_POSITIVE_RATIO = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/ROI_POSITIVE_RATIO")

    	# Pooled ROIs
    	POOL_SIZE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/POOL_SIZE")
    	MASK_POOL_SIZE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/MASK_POOL_SIZE")

    	# Shape of output mask
    	# To change this you also need to change the neural network mask branch
    	MASK_SHAPE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/MASK_SHAPE")

    	# Maximum number of ground truth instances to use in one image
    	MAX_GT_INSTANCES = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/MAX_GT_INSTANCES")

    	# Bounding box refinement standard deviation for RPN and final detections.
    	RPN_BBOX_STD_DEV = np.array(rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/RPN_BBOX_STD_DEV"))
    	BBOX_STD_DEV = np.array(rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/BBOX_STD_DEV"))

    	# Max number of final detections
    	DETECTION_MAX_INSTANCES = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/DETECTION_MAX_INSTANCES")

    	# Minimum probability value to accept a detected instance
    	# ROIs below this threshold are skipped
    	DETECTION_MIN_CONFIDENCE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/DETECTION_MIN_CONFIDENCE")

    	# Non-maximum suppression threshold for detection
    	DETECTION_NMS_THRESHOLD = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/DETECTION_NMS_THRESHOLD")

    	# Learning rate and momentum
    	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    	# weights to explode. Likely due to differences in optimzer
    	# implementation.
    	LEARNING_RATE = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/LEARNING_RATE")
    	LEARNING_MOMENTUM = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/LEARNING_MOMENTUM")

    	# Weight decay regularization
    	WEIGHT_DECAY = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/WEIGHT_DECAY")

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
    	USE_RPN_ROIS = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/USE_RPN_ROIS")

    	# Train or freeze batch normalization layers
    	#     None: Train BN layers. This is the normal mode
    	#     False: Freeze BN layers. Good when using a small batch size
    	#     True: (don't use). Set layer in training mode even when inferencing
    	TRAIN_BN = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/TRAIN_BN")  # Defaulting to False since batch size is often small

    	# Gradient norm clipping
    	GRADIENT_CLIP_NORM = rospy.get_param("/mask_rcnn/mask_rcnn_model/mask_rcnn_configs/GRADIENT_CLIP_NORM")

    	#NAME = "tableware"
    	#IMAGES_PER_GPU = 1
    	#GPU_COUNT = 1
    	#NUM_CLASSES = 1 + 9  # COCO has 80 classes (1+80)
    	#BACKBONE = "mobilenetv1"
    	#BACKBONE_STRIDES = [4, 8, 16, 32, 64] #ResNet
    	##BACKBONE_STRIDES = [2, 4, 8, 16, 32]
    	##RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256) #ResNet
    	#RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)
    	#MINI_MASK_SHAPE = (56, 56) #ResNet
    	##MINI_MASK_SHAPE = (28, 28)
    	##IMAGE_MIN_DIM = 400
    	#IMAGE_MAX_DIM = 512
    	##TRAIN_ROIS_PER_IMAGE = 200 #ResNet
    	##TRAIN_ROIS_PER_IMAGE = 128
    	# Use Multiprocessing in MaskRCNN.train()
    	#USE_MULTIPROCESSING = True
	#DETECTION_MIN_CONFIDENCE = 0.85

 
    config = InferenceConfig()
    config.display()
    
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs_path)
    model.load_weights(weights_file, by_name=True)

    while not rospy.is_shutdown() and rosgraph.is_master_online():
        print("Waiting for image\n")
        try:
            img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage, timeout=1)
            img_np_arr = np.fromstring(img.data, np.uint8)
            encoded_img = cv2.imdecode(img_np_arr, 1)
            #encoded_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB) 
            results = model.detect([cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)], verbose=1)
            #results = model.detect([encoded_img], verbose=1)

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

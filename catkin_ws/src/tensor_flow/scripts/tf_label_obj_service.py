#!/usr/bin/env python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import sys, os
import rospy
import cv2
from tensor_flow.srv import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def read_tensor_with_opencv(cv_image, input_height=299, input_width=299, input_mean=0, input_std=255):
  image_decoded = cv_image 
  image_decoded = cv2.resize(image_decoded, dsize=(299, 299), interpolation = cv2.INTER_CUBIC)
  np_image_data = np.asarray(image_decoded)
  float_caster = tf.cast(np_image_data,tf.float32)
  #np_final = np.expand_dims(np_image_data,axis=0)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander,[input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

  sess = tf.Session()
  result = sess.run(normalized)
  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def callbackImageLabel(req):
  global graph, labels, input_operation, output_operation 

  bridge =CvBridge()
  try:
      cv_image = bridge.imgmsg_to_cv2(req.image, "bgr8")
  except CvBridgeError as e:
      print(e)
  #cv2.imshow("Image window", cv_image)
  #cv2.waitKey(3)

  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  #input_layer = "Placeholder"
  #output_layer = "final_result"
  
  t = read_tensor_with_opencv(
          cv_image,
          input_height=input_height,
          input_width=input_width,
          input_mean=input_mean,
          input_std=input_std)
  print ("test")
  
  #input_name = "import/" + input_layer
  #output_name = "import/" + output_layer
  #input_operation = graph.get_operation_by_name(input_name)
  #output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1] #[::-1]=decreasing order, [-5:]=only the first 5 elements
  for i in top_k:
    print(labels[i], results[i])

  #print(top_k)

  return image_srvResponse(labels[top_k[0]])


if __name__ == "__main__":
  global graph, labels, input_operation, output_operation 

  rospy.init_node('tensor_flow')

  rospy.Service('/tensor_flow/image', image_srv, callbackImageLabel)
  
  filePath = os.path.dirname(os.path.abspath(__file__))
  file_path_h, file_path_t = os.path.split(filePath)

  model_file = file_path_h + "/graph/output_graph.pb"
  label_file = file_path_h + "/graph/output_labels.txt"
  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)
  labels = load_labels(label_file)
  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  rospy.spin()


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

import sys
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

def read_tensor_with_opencv(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
  image_decoded = cv2.imread("/home/biorobotica/docker_volumen/JUSTINA/catkin_ws/src/tensor_flow/scripts/diente-deleon.jpg")
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
  a = 5
  try:
      cv_image = bridge.imgmsg_to_cv2(req, "bgr8")
  except CvBridgeError as e:
      print(e)
    
  cv2.imshow("Image window", cv_image)
  cv2.waitKey(3)
  return image_srvResponse("response")


if __name__ == "__main__":
  #rospy.init_node('tensor_flow')

  #rospy.Service('/tensor_flow/image', image_srv, callbackImageLabel)

  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "/home/biorobotica/docker_volumen/JUSTINA/catkin_ws/src/tensor_flow/graph/output_graph.pb"
  label_file = "/home/biorobotica/docker_volumen/JUSTINA/catkin_ws/src/tensor_flow/graph/output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  #t = read_tensor_from_image_file(
  #    file_name,
  #    input_height=input_height,
  #    input_width=input_width,
  #    input_mean=input_mean,
  #    input_std=input_std)

  t = read_tensor_with_opencv(
          file_name,
          input_height=input_height,
          input_width=input_width,
          input_mean=input_mean,
          input_std=input_std)
  print ("test")

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  for i in top_k:
    print(labels[i], results[i])

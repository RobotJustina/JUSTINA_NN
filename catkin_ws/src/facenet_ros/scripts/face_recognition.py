import pickle
import os
import math

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet

class Face:
    def __init__(self):
        self.name = None
        self.probability = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

class Recognition:
    def __init__(self, facenet_model, data_dir, face_crop_size=160, face_crop_margin=32, minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709, gpu_memory_fraction=0.3, detect_multiple_faces=True):
        self.detect = Detection(face_crop_size=face_crop_size, face_crop_margin=face_crop_margin, minsize=minsize, threshold=threshold, factor=factor, gpu_memory_fraction=gpu_memory_fraction, detect_multiple_faces=detect_multiple_faces)
        dataset = facenet.get_dataset(data_dir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        self.encoder = Encoder(facenet_model=facenet_model)
        with self.encoder.sess.as_default():
            self.identifier = Identifier(self.encoder.generate_embeddings(paths, 20, face_crop_size), labels, class_names)

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image, debug = False):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name, face.probability = self.identifier.identify(face)

        return faces

class Identifier:
    def __init__(self, db_emb, labels, class_names):
        self.db_emb = db_emb
        self.labels = labels
        self.class_names = class_names

    def identify(self, face):
        if face.embedding is not None:
            best_class_index = 0
            best = 10.0
            for db_index in range(len(self.db_emb)): 
                dist = np.sqrt(np.sum(np.square(np.subtract(self.db_emb[db_index,:], face.embedding))))
                if dist < best:
                    best = dist
                    best_class_index = db_index
            if best < 0.95:
                return self.class_names[self.labels[best_class_index]], best
            else:
                return "Unknown", abs(0.95 - best)

class Encoder:
    def __init__(self, facenet_model):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def generate_embeddings(self, paths, batch_size, image_size):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i+1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    gpu_memory_fraction = 0.3
    detect_multiple_faces = True

    def __init__(self, face_crop_size=160, face_crop_margin=32, minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709, gpu_memory_fraction=0.3, detect_multiple_faces=True):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor
        self.gpu_memory_fraction = gpu_memory_fraction
        self.detect_multiple_faces = detect_multiple_faces

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                return align.detect_face.create_mtcnn(self.sess, None)

    def find_faces(self, image):
        faces = []

        with self.sess.as_default():
            bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        
        if self.detect_multiple_faces:
            for bb in bounding_boxes:
                face = Face()
                face.container_image = image
                face.bounding_box = np.zeros(4, dtype=np.int32)

                img_size = np.asarray(image.shape)[0:2]
                face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
                face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
                face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
                face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
                cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
                face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

                faces.append(face)
        elif len(bounding_boxes) > 0:
            img_size = np.asarray(image.shape)[0:2]
            det = bounding_boxes[:,0:4]
            bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
            img_center = img_size / 2
            offsets = np.vstack([ (det[:,0] + det[:,2])/2 - img_center[1], (det[:,1] + det[:,3])/2 - img_center[0] ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(det[index,:][0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(det[index,:][1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(det[index,:][2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(det[index,:][3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)

        return faces

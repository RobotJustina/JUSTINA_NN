#!/usr/bin/env python

import face_recognition
import cv2
import os
import os.path
import rospy
import rosgraph
import numpy as np

from face_recognition.face_recognition_cli import image_files_in_folder
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from face_recog.msg import *
from face_recog.srv import *



def train_Images():
    train_dir = "scripts/faces"
	# Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                Faces.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                names.append(class_dir)

def face_recognition_callback(req):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    ret, frame = bridge.imgmsg_to_cv2(req.imageBGR, "bgr8")

    if process_this_frame:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        labels = []
        distances = []

        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(Faces, face_encoding, tolerance)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]

            labels.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, labels):
        bounding_box = [Point(top, right, 0), Point(bottom, left, 0)]
        face_centroid = Point((top + bottom)/2, (right + left)/2, 0)
        face_class = VisionFaceObject(id = name, confidence = 0, face_centroid=face_centroid, bounding_box=bounding_box)
        recog_faces.append(face_class)
    
    return FaceRecognitionResponse(VisionFaceObjects(h, recog_faces))





def main():
    rospy.init_node('face_recognition_node')
    rate = rospy.Rate(30)

    global labels
    global process_this_frame
    global distances
    global Faces
    global names 

    Faces = []
    names = []

    process_this_frame = True
    
    verbose = True
    tolerance = 0.55

    train_Images()

    bridge = CvBridge()

    s = rospy.Service('face_recog/faces', FaceRecognition, face_recognition_callback)

    while not rospy.is_shutdown() and rosgraph.is_master_online():
        print("Waiting for image\n")
        try:
            img = rospy.wait_for_message("/usb_cam/image_raw", Image, timeout = 1)
            image_color = CvBridge().imgmsg_to_cv2(img,'bgr8')
            
            if process_this_frame:
                face_locations = face_recognition.face_locations(image_color)
                face_encodings = face_recognition.face_encodings(image_color,face_locations)

                labels = []
                distances =[]

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(Faces, face_encoding, tolerance)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = names[first_match_index]

                    labels.append(name)
            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, labels):
                #Draw a box around the face
                cv2.rectangle(image_color, (left, top), (right, bottom), (0,0,255),2)

                #Draw a label with a name below the face
                cv2.rectangle(image_color, (left, bottom -35), (right, bottom), (0,0,255), cv2.FILLED)
                font= cv2. FONT_HERSHEY_DUPLEX
                cv2.putText(image_color, name, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', image_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except rospy.ROSException as e:
            if 'timeout exceeded' in e.message:
                continue  # no new waypoint within timeout, looping...
            else:
                raise e

        rate.sleep()
if __name__=='__main__':
    main()

    
    




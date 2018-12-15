#!/usr/bin/env python

import face_recognition
import cv2
import os
import os.path
import glob
import re
import rospy
import rosgraph
import numpy as np
import rospkg
import shutil

from face_recognition.face_recognition_cli import image_files_in_folder
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from justina_nn_msgs.msg import *
from justina_nn_msgs.srv import *
from std_msgs.msg import Empty

def sorted_nicely(strings):
    return sorted(strings, key=natural_sort_key)

def natural_sort_key(key):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def load_Images():

    global Faces
    global names
    global train_dir 
    global face_bounding_boxes
    global verbose
    
	# Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) > 1:
                img_size = np.asarray(image.shape)[0:2]
                det = []
                for (top, right, bottom, left) in face_bounding_boxes:
                    det.append(np.array([left, bottom, right, top]))
                det = np.array(det)
                bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0] + det[:,2])/2 - img_center[1], (det[:,1] + det[:,3])/2 - img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                Faces.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[index])
                names.append(class_dir)
            elif len(face_bounding_boxes) == 1:
                # Add face encoding for current image to the training set
                Faces.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                names.append(class_dir)
            else:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
    
def train_faces_callback(req):
    try:
        cv_image = bridge.imgmsg_to_cv2(req.imageBGR, 'bgr8')
    except CvBridgeError as e:
        print(e)
    train_new_face(cv_image, req.id)
    return FaceRecognitionResponse()

def train_new_face(image, name):
    global train_dir
    global labels
    global process_this_frame
    global distances
    global Faces
    global names
    global face_bounding_boxes

    print('Training person' + name)
    #id = msg.data
    id = name

    path = train_dir + "/" + id
    try:  
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
    reg_exp = path + "/" + id + "_[0-9]*.jpg"
    result = sorted_nicely( glob.glob(reg_exp))
    if (len(result) == 0):
        name_image = id + "_0.jpg";
    else:
        last_result = result[-1]
        number = re.search( path + "/" + id + "_([0-9]*).jpg",last_result).group(1)
        name_image = id + "_%i.jpg"%+(int(number)+1)
    
    
    print name_image

    cv2_image = image
    cv2.imwrite(os.path.join(path, name_image), cv2_image) 
    new_image = face_recognition.load_image_file(path + "/" + name_image)
    face_bounding_boxes = face_recognition.face_locations(new_image)
        
    if len(face_bounding_boxes) > 1:
        img_size = np.asarray(image.shape)[0:2]
        det = []
        for (top, right, bottom, left) in face_bounding_boxes:
            det.append(np.array([left, bottom, right, top]))
        det = np.array(det)
        bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0] + det[:,2])/2 - img_center[1], (det[:,1] + det[:,3])/2 - img_center[0] ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
        Faces.append(face_recognition.face_encodings(new_image, known_face_locations=face_bounding_boxes)[index])
        names.append(id)
    elif len(face_bounding_boxes) == 1:
        # Add face encoding for current image to the training set
        Faces.append(face_recognition.face_encodings(new_image, known_face_locations=face_bounding_boxes)[0])
        names.append(id)
    else:
        # If there are no people (or too many people) in a training image, skip the image.
        if verbose:
            print("Image {} not suitable for training: {}".format(path + "/" + name_image, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))

    

def face_recognition_callback(req):
    global train_dir
    global labels
    global process_this_frame
    global distances
    global Faces
    global names
    global bridge
    global face_encodings
    global face_locations

    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()

    try:
        frame = bridge.imgmsg_to_cv2(req.imageBGR, 'bgr8')
    except CvBridgeError as e:
        print(e)

    recog_faces = []
    if process_this_frame:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        labels = []
        distances = []
        
        index_face = 0
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(Faces, face_encoding, tolerance)
            name = "Unknown"
            face_distance = 0

            if len(matches) > 0:
                face_distances = face_recognition.face_distance(Faces, face_encoding)
                if True in matches:
                    first_match_index = np.argmin(face_distances)
                    name = names[first_match_index]
                else:
                    first_match_index = np.argmin(face_distances)
                face_distance = face_distances[first_match_index]
                if(req.id == '' or name == req.id.replace("_"," ")):
                    face_distance = 1 - round(face_distance,2)
                    (top, right, bottom, left) = face_locations[index_face]
                    bounding_box = [Point(left, top, 0), Point(right, bottom, 0)]
                    face_centroid = Point((right + left)/2, (top + bottom)/2, 0)
                    face_class = VisionFaceObject(id = name, confidence = face_distance, face_centroid=face_centroid, bounding_box=bounding_box)
                    recog_faces.append(face_class)
                index_face+=1
    
    return FaceRecognitionResponse(VisionFaceObjects(h, recog_faces))

def clear_faces_callback(msg):
    global clear_bd
    clear_bd = True

def clear_face_id_callback(msg):
    global clear_face_id
    global face_clear_id
    clear_face_id = True
    face_clear_id = msg.data



def main():
    rospy.init_node('face_recognition_lib_node')
    rate = rospy.Rate(30)
    rospack = rospkg.RosPack()
    global train_dir
    #train_dir = rospack.get_path('face_recog') + "/faces"
    train_dir = rospy.get_param("~train_dir")

    global labels
    global process_this_frame
    global distances
    global Faces
    global names 
    global bridge
    global face_locations
    global face_encodings
    global verbose
    global tolerance
    global clear_face_id
    global clear_bd
    global face_clear_id

    clear_face_id = False
    clear_bd = False
    face_clear_id = ""
    face_recog_available = True

    try:  
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % train_dir)
    else:  
        print ("Successfully created the directory %s " % train_dir)

    Faces = []
    names = []

    process_this_frame = True
    
    verbose = True
    tolerance = 0.55

    load_Images()

    bridge = CvBridge()

    s = rospy.Service('face_recognizer/faces', FaceRecognition, face_recognition_callback)
    st = rospy.Service('face_recognizer/train_face', FaceRecognition, train_faces_callback)
    rospy.Subscriber("face_recognizer/clear_faces", Empty, clear_faces_callback)
    rospy.Subscriber("face_recognizer/clear_face_id", String, clear_face_id_callback)
    #rospy.Subscriber("face_recog/train_online", String, callbackTrain)

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
                        face_distances = face_recognition.face_distance(Faces, face_encoding)
                        face_distance = face_distances[first_match_index]
                        face_distance = 1 - round(face_distance,2) 

                    labels.append(name)
                    
            process_this_frame = not process_this_frame


            for (top, right, bottom, left), name in zip(face_locations, labels):
                #Draw a box around the face
                cv2.rectangle(image_color, (left, top), (right, bottom), (0,0,255),2)

                #Draw a label with a name below the face
                cv2.rectangle(image_color, (left, bottom -35), (right, bottom), (0,0,255), cv2.FILLED)
                font= cv2. FONT_HERSHEY_DUPLEX
                cv2.putText(image_color, name, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)
                cv2.putText(image_color, str(face_distance), (right - 50, bottom -6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', image_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except rospy.ROSException as e:
            pass
        if clear_bd or clear_face_id:
            print('Trying to clean data base')
            if os.path.exists(train_dir):
                if clear_bd:
                    shutil.rmtree(train_dir, ignore_errors=True)
                    Faces = []
                    names = []
                if clear_face_id:
                    person_dir = train_dir + "/" + face_clear_id
                    if os.path.exists(person_dir):
                        shutil.rmtree(person_dir, ignore_errors=True)
                        index = 0
                        while index < len(names):
                            if names[index]==face_clear_id:
                                names.pop(index)
                                Faces.pop(index)
                            else:
                                index = index + 1
            clear_bd = False
            clear_face_id = False

        
        rate.sleep()
if __name__=='__main__':
    main()

    
    




import face_recognition
import cv2
import os
import os.path
from face_recognition.face_recognition_cli import image_files_in_folder

video_capture = cv2.VideoCapture(0)

Faces = []
names = []



train_dir = "faces"
verbose = True
tolerance = 0.55

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

# Initialize some variables
labels = []
process_this_frame = True
distances = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = frame[:, :, ::-1]	
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        labels = []
	distances =[]
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(Faces, face_encoding, tolerance)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
		#face_distances = face_recognition.face_distance(Faces, face_encoding)

            labels.append(name)
	    #distances.append(face_distances)
	    #print("The test image has a distance of {} ".format(face_distances))

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, labels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #top *= 4
        #right *= 4
        #bottom *= 4
        #left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	#cv2.putText(frame, face_distances.tostring(), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()





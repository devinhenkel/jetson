import os
import glob

import face_recognition
import cv2
import numpy as np
from datetime import datetime, timedelta
import platform
import pickle
import pyttsx3


def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=2):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

if running_on_jetson_nano():
    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
else:
    # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
    # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
    video_capture = cv2.VideoCapture(0)

# Our list of known face encodings and a matching list of metadata about each face.
known_face_images = []
known_face_encodings = []
known_face_names = []

###path = '/home/devinhenkel/doorcam/faces'
##path = ''
##
##for file in glob.glob(os.path.join(path, '*.jpg')):
##    print('Encoding: ' + file)
##    temp_image = face_recognition.load_image_file(file)
##    temp_face_locations = face_recognition.face_locations(temp_image)
##    print(temp_face_locations)
##    temp_face_encoding = face_recognition.face_encodings(temp_image)[0]
##    known_face_encodings.append(temp_face_encoding)
##    known_face_names.append(file.split('.')[0])

def load_known_faces():
    global known_face_encodings, known_face_names

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_names = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass
    
load_known_faces()
    
print(known_face_names)
    
 # Initialize some variables
face_locations = []
face_encodings = []
face_names = []
seen_face_names = []
seen_face_times = []
engine = pyttsx3.init()
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name not in seen_face_names:
                seen_face_names.append(name)
                seen_face_times.append(datetime.now())
                engine.say("Hello, "+name)
                engine.runAndWait()
            else:
                face_index = seen_face_names.index(name)
                if datetime.now() - seen_face_times[face_index] > timedelta(minutes=5):
                    engine.say("Welcome back, "+name)
                    engine.runAndWait()
                    seen_face_times[face_index] = datetime.now()
                else:
                    seen_face_times[face_index] = datetime.now()
                    

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



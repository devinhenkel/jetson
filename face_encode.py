import os
import glob

import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle


# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_names = []


def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_names]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_names

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_names = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass


#path = '/home/devinhenkel/doorcam/faces'
path = ''

for file in glob.glob(os.path.join(path, '*.jpg')):
    print('Encoding: ' + file)
    temp_image = face_recognition.load_image_file(file)
    temp_face_locations = face_recognition.face_locations(temp_image)
    print(temp_face_locations)
    temp_face_encoding = face_recognition.face_encodings(temp_image)[0]
    known_face_encodings.append(temp_face_encoding)
    known_face_names.append(file.split('.')[0])
    
print(known_face_names)

save_known_faces()
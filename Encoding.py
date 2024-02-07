import os
import face_recognition
import cv2
import numpy as np
import pickle

path='Studentimages/'
encodings_file = 'encodings.pickle'
images = []
classnames = []
mylist = os.listdir(path) # a list to store all the paths of student images 


#User defined function to load our pickle file in program 
def load_encodings(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return [], []

#User defined function to save the encodings in our pickle file
def save_encodings(encodings, class_names, filename):
    with open(filename, 'wb') as f:
        pickle.dump((encodings, class_names), f)


existing_encodings, existing_classnames = load_encodings(encodings_file)


#traversing our image list to build classes
for it in mylist:
    if os.path.splitext(it)[0] in existing_classnames:
        print(f"Skipping {it} as it is already encoded.") #if there is already an encoding for the image we skip it
        continue

    curImg = cv2.imread(os.path.join(path, it))
    if curImg is not None:
        print(f"Encoding the data for {it}....")
        images.append(curImg)
        class_name = os.path.splitext(it)[0]
        classnames.append(class_name)
    else: #in case the image is not in suitable format/corrupt
        print(f"Image {it} not loaded properly.")

encode_list = []
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded = face_recognition.face_encodings(img)
    if len(encoded) > 0:
        encode_list.append(encoded[0])
    else:
        print(f"No face found in the image.")

final_encodings = existing_encodings + encode_list  #merging the new encodings  with prev
final_classnames = existing_classnames + classnames #merging the new classnames with prev

save_encodings(final_encodings, final_classnames, encodings_file)
print("Encodings updated and saved successfully.")

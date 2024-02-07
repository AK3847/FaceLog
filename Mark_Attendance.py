import face_recognition
import cv2
import numpy as np
from datetime import datetime
import openpyxl
import pickle

encodings_file = 'encodings.pickle'

#user defined function to read the pickel file
def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#storing the encoding list and classnames from the pickel file
encode_list, classnames=load_encodings(encodings_file)


#user defined function to access the Student Attendance Excel sheet
def write_to_excel(student_name):
    wb=openpyxl.load_workbook('Student Attendance.xlsx')
    #wb=openpyxl.load_workbook('Student Attendance.xlsx')
    sheet=wb.active
    current_date_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row=[student_name, current_date_time]
    sheet.append(row)
    wb.save('Student Attendance.xlsx')
    wb.close()

print("Starting the camera to capture your pic...")
print("Press spacebar to capture your image..")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (217,136, 31), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):# the capturing stops when user presses spacebar
        cv2.imwrite('captured_image.jpg', frame) #storing the captured frame as a jpg
        break

cap.release()
cv2.destroyAllWindows()

captured_image = cv2.imread('captured_image.jpg') 
captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
captured_image_encoding = face_recognition.face_encodings(captured_image_rgb)

#if else block to check if we found a face or not...
if len(captured_image_encoding) > 0:
    captured_encoding = captured_image_encoding[0]

    results = face_recognition.compare_faces(encode_list, captured_encoding)

    for i, result in enumerate(results):
        if result:
            student_name = classnames[i]
            print(f"Match found! The captured image is of {student_name}.")
            write_to_excel(student_name)
            break
    else:
        print("No match found.")

else:
    print("No face found in the captured image.")

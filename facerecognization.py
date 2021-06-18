import cv2
import numpy as np
import pywhatkit as pw
import time
import smtplib
import os

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = '/home/gunasekhar/Desktop/ws/i1/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples of face 1 Completed")

cap.release()


cap = cv2.VideoCapture(0)
count1 = 0

while True:

    ret1, frame1 = cap.read()
    if face_extractor(frame1) is not None:
        count1 += 1
        face1 = cv2.resize(face_extractor(frame1), (200, 200))
        face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path1 = '/home/gunasekhar/Desktop/ws/i2/' + str(count1) + '.jpg'
        cv2.imwrite(file_name_path1, face1)

        # Put count on images and display live count
        cv2.putText(face1, str(count1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face1)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count1 == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples face 2 is Completed")



cap.release()

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = '/home/gunasekhar/Desktop/ws/i1/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

gunasekhar_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train

gunasekhar_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")







# Get the training data we previously made
data_path1 = '/home/gunasekhar/Desktop/ws/i2/'
onlyfiles1 = [g for g in listdir(data_path1) if isfile(join(data_path1, g))]

# Create arrays for training data and labels
Training_Data1, Labels1 = [], []



# Open training images in our datapath
# Create a numpy array for training data
for k, files1 in enumerate(onlyfiles1):
    image_path1 = data_path1 + onlyfiles1[k]
    images1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    Training_Data1.append(np.asarray(images1, dtype=np.uint8))
    Labels1.append(k)

# Create a numpy array for both training data and labels
Labels1 = np.asarray(Labels1, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

other_model= cv2.face_LBPHFaceRecognizer.create()
# Let's train

other_model.train(np.asarray(Training_Data1), np.asarray(Labels1))
print("Other Model trained sucessefully")


import cv2
import numpy as np
import os


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = gunasekhar_model.predict(face)
        results1= other_model.predict(face)
        # harry_model.predict(face)
        if results1[1]<500:
            confidence1 = int( 100 * (1 - (results1[1])/400) )
            display_string = str(confidence1) + '% Confident it is User'
          
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
    
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Guna Sekhar", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            t=time.localtime()
            pw.sendwhatmsg('+91 phone number','Guna Sekhar face detected',t[3],t[4]+2)
            import smtplib
  
            # creates SMTP session
            s = smtplib.SMTP('smtp.gmail.com', 587)
  
            # start TLS for security
            s.starttls()
  
            # Authentication
            s.login("senderemailid", "password")
  
            # message to be sent
            message = "This is face of Guna Sekhar"
  
            # sending the mail
            s.sendmail("senderemailid","receiveremailid", message)
  
            # terminating the session
            s.quit()
            break
         
        else:
            if confidence1 > 80:
                cv2.putText(image, "Hey other", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                os.system('terraform init')
                os.system('terraform apply --auto-approve')
                break
            else:
                cv2.putText(image, "Model not trained on your face", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )
                pass
            
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()     



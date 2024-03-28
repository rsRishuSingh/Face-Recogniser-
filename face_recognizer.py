import cv2 as cv
import numpy as np

def rescaleFrame(frame):
    
    dimensions = (600, 600)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

people = ['Akshay','Deepika','katrina','salmankhan','shahrukhkhan']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('face_trained.yml')

img  = cv.imread(r"C:\Users\Rishu Singh\OneDrive\Documents\coding\OpenCv\img_validation\katrina\img4.jpeg")
# img = rescaleFrame(img)
# cv.imshow(" img", img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces_react = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_react:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recogniser.predict(faces_roi)
    print("label : ",people[label])
    print("confidence : ",confidence)

    cv.putText(img, str(people[label]), (100,100), cv.FONT_HERSHEY_COMPLEX,1, (0,255,0),thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

img = rescaleFrame(img)
cv.imshow("detected img", img)
cv.waitKey(0)
import cv2 as cv 
import numpy as np 

def rescaleFrame(frame):
    width = 600
    height = 600

    dimensions = (width, height)

    return cv.resize(frame, dimensions,  interpolation=cv.INTER_AREA)

img = cv.imread("image/parents.jpg")
img = rescaleFrame(img)
cv.imshow("Parents image ", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray parents ",gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print("NUmber of faces found ",len(faces_react))
print("NUmber of faces found ",faces_react)

for (x,y,w,h) in faces_react:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv.imshow("Detected image ",img)
cv.waitKey(0)


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:54:08 2017

@author: Ajay
"""
#Face Detection App

#Importing the libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Defining the function that will do the detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5) #faces are 4 element tuple(x,y,width,height)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255),2)# for each of the faces we draw a rectangle around them
        roi_gray= gray[y:y+h,x:x+w] #We are restricting the region to the rectangle we just foung for the eye
        roi_color=frame[y:y+h,x:x+w]#region of interest for the color
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1 , 3)#detecting the eyes in the face
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (255,0,0),2)
    return frame

#Doing the detection using the webcam
video_capture= cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas= detect(gray,frame)
    cv2.imshow('Face Detector',canvas)
    if cv2.waitKey(1) &0xff==ord('q'):
        break
    

video_capture.release()
cv2.destroyAllWindows()






    
            
        
    
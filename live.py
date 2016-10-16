import sys
sys.path.append("/home/pi/.virtualenvs/cv/lib/python2.7/site-packages")

import numpy as np
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import argparse
import glob

import pickle
import sys
from histogram.rgbhistogram import RGBHistogram


# load it again
fid = open('model.pkl', 'rb')
model = pickle.load(fid)


target = ["Cafe De La India", "Cruz De Marta", "Flamboyan", "Helicornia", "Nothing"  ]
data = []
# Grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)
desc = RGBHistogram([8, 8, 8])




#Camera Set-UP
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.hflip = True
LiveCapture = PiRGBArray(camera, size = (640, 480))

#Camera Warm-Up
time.sleep(0.5)




#Capture Live Video and save each frame to image
for frame in camera.capture_continuous(LiveCapture, format = "bgr", use_video_port = True):


    image = np.array(frame.array)

    #comparing the image to the model
    targetImage = image
    mask = np.zeros(targetImage.shape[:2], dtype = "uint8")
    (cX, cY) = (targetImage.shape[1] // 2, targetImage.shape[0] //2)
    r = int(round(cX/2))
    cv2.circle(mask, (cX, cY), r, 255, -1)
    features = desc.describe(targetImage, mask)
    
    

    # redict what type of flower the image is YAY!
    flower = le.inverse_transform(model.predict(features))[0]
    print(format(flower.upper()))
    
    
       
   
    

# Perform the actual resizing of the image
    r = 480.0 / image.shape[1]
    dim = (480, int(image.shape[0] * r))

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    if format(flower) == "Nothing":
        
        camera.annotate_text = ""
    
    elif format(flower) == "Flamboyan":

        
    

    #HSV -----------------------------------------------------------------
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        
        GreenLower = np.array([18, 146, 164], dtype = "uint8")
        GreenUpper = np.array([26, 255, 255], dtype = "uint8")
    #-------------------------------------------------------------
        Green = cv2.inRange(hsv, GreenLower, GreenUpper)
        Green = cv2.erode(Green, None, iterations = 2)
        Green = cv2.dilate(Green, None, iterations = 2)
    #------------------------------------------------------------------------
        image2 = cv2.GaussianBlur(Green, (31, 31), 0)
        canny = cv2.Canny(image2, 30, 100)


        


    #Find the Contour in image
        (_ , contour, hieratchy) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        max = 0
        cnt =[]
        for c in contour:
            if cv2.contourArea(c) > max: 
                max = cv2.contourArea(c) 
                cnt = c
        blue = (255, 0, 0)
        

        if len(cnt) >1:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(resized,(x/2,y/2),(x+w*3,y+h*3), blue, 2)

        camera.annotate_text = "Famboyan"
#--------------------------------------------------



        resized2 = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)

    
    elif format(flower) == "Cruz De Marta":

        
    

    #HSV -----------------------------------------------------------------
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        
        GreenLower = np.array([146, 02, 137], dtype = "uint8")
        GreenUpper = np.array([179, 125, 255], dtype = "uint8")
    #-------------------------------------------------------------
        Green = cv2.inRange(hsv, GreenLower, GreenUpper)
        Green = cv2.erode(Green, None, iterations = 2)
        Green = cv2.dilate(Green, None, iterations = 2)
    #------------------------------------------------------------------------
        image2 = cv2.GaussianBlur(Green, (31, 31), 0)
        canny = cv2.Canny(image2, 30, 100)


        


    #Find the Contour in image
        (_ , contour, hieratchy) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        max = 0
        cnt =[]
        for c in contour:
            if cv2.contourArea(c) > max: 
                max = cv2.contourArea(c) 
                cnt = c
        blue = (255, 0, 0)
        

        if len(cnt) >1:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(resized,(x/2,1+y/2),(x+w*2,1+y+h*2), blue, 2)

        camera.annotate_text = "Cruz De Marta"
#--------------------------------------------------



        resized2 = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)
        
    elif format(flower) == "Cafe De La India":

        
    

    #HSV -----------------------------------------------------------------
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        
        GreenLower = np.array([5, 0, 135], dtype = "uint8")
        GreenUpper = np.array([183, 108, 255], dtype = "uint8")
    #-------------------------------------------------------------
        Green = cv2.inRange(hsv, GreenLower, GreenUpper)
        Green = cv2.erode(Green, None, iterations = 2)
        Green = cv2.dilate(Green, None, iterations = 2)
    #------------------------------------------------------------------------
        image2 = cv2.GaussianBlur(Green, (31, 31), 0)
        canny = cv2.Canny(image2, 30, 100)


        


    #Find the Contour in image
        (_ , contour, hieratchy) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        max = 0
        cnt =[]
        for c in contour:
            if cv2.contourArea(c) > max: 
                max = cv2.contourArea(c) 
                cnt = c
        blue = (255, 0, 0)
        

        if len(cnt) >1:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(resized,(x/2,y/2),(x+w*2,y+h*2), blue, 2)

        camera.annotate_text = "Cafe De La India"
#--------------------------------------------------



        resized2 = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)


    
    cv2.imshow("Live Video", resized)
    

    LiveCapture.truncate(0)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()


        

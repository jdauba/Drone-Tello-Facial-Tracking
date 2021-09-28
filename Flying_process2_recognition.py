# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:39:24 2021
@author: jerom
"""
## Tracking corps si perte visage

import cv2
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import tello
from time import sleep
import sys, os
sys.path.append('C:/Users/jerom/Desktop/Projet Tello/')
from face_tracking_test_recognitionV2 import detect_track_face

# Drone init :
## TELLO NAME : 62A694
me = tello.Tello()
me.connect()
print(me.get_battery())

# Start stream :
me.streamon()
print("INFO : stream ON")

#Init image object :
image_read = me.get_frame_read()

#Drone take off
me.takeoff()
print("Take off OK")
me.move_up(40)
print("FLying...")

#Main loop :
while True :
    # Image visualisation
    image = image_read.frame
    image = cv2.resize(image, (720, 480))
    
    # Face detection function :
    detect_track_face(me, image, track=True)
    
    #Visualization :
    cv2.imshow('Vizualising Drone Target...', image)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        print("Landing in progress...")
        me.land()
        print("Landed mildly successfully !")
        print("Cutting stream...")
        me.streamoff()
        cv2.destroyAllWindows()
        break



    
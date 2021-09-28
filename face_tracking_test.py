# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:24:13 2021

@author: jerom
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import tello
from time import sleep
import sys, os
sys.path.append('C:/Users/jerom/Desktop/Projet Tello/')
from objcenter import ObjCenter
from pid import PID
#import imutils


#Init face center
face_center = ObjCenter(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Init PID :
left_right_pid = PID(kP=0.6, kI=0.0001, kD=0.1)
up_down_pid = PID(kP=0.6, kI=0.0001, kD=0.1)
left_right_pid.initialize()
up_down_pid.initialize()

#Forward PID :
wref = 65
forward_pid = PID(kP=1.8, kI=0.0001, kD=0.1)
forward_pid.initialize()
K_forward = 1.5
K_backward = 0.5
# Max speed :
max_speed_threshold = 120

#Capture video :
capture = True

if capture :
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Videos/Drone8.avi',fourcc, 20.0, (720,480))

def detect_track_face(me, frame, track):
    H,W,_ = frame.shape
    # On crée un "centre de frame" légèrement décalé vers le haut de l'image
    centerX = W//2
    centerY = H//3
    frame_center = (centerX, centerY)
    
    #Draw a red circle at the center of a frame:
    cv2.circle(frame, center=(centerX,centerY), radius=5, color=(0,0,255), thickness=-1)
    
    # Drawing rectangle for detected face :
    objectLoc = face_center.update(frame, frameCenter=None)
    ((objX,objY), rect, d) = objectLoc
    
    if objX is None or objY is None :
        objX, objY = centerX, centerY

    try:
        x, y, w, h = rect
        #print("WIDTH :", w)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        face_detected = True
    except:
        print("Aucun visage détecté")
        face_detected = False
        pass
    
    #Drawing an arrow from center to face (represente the drone's displacement vector)

    cv2.arrowedLine(frame, frame_center, (objX, objY), color=(0,255,0), thickness=2)
    
    if track :
        if not face_detected :
            # Drone don't move
            me.send_rc_control(0, 0, 0, 0)
        else :
            left_right_error = centerX - objX
            left_right_update = left_right_pid.update(left_right_error, sleep=0)

            up_down_error = centerY - objY
            up_down_update = up_down_pid.update(up_down_error, sleep=0)
            
            forward_error = wref - w
            if forward_error > 0 :
                forward_error = K_forward * forward_error
            else:
                forward_error = K_backward * forward_error
            forward_update = forward_pid.update(forward_error, sleep=0)

            cv2.putText(frame, f"X Error: {left_right_error} PID: {left_right_update:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Y Error: {up_down_error} PID: {up_down_update:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Forward Error: {forward_error} PID: {forward_update:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Target name : searching...", (20, 450), cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255), 2, cv2.LINE_AA)

            if left_right_update > max_speed_threshold:
                left_right_update = max_speed_threshold
            elif left_right_update < -max_speed_threshold:
                left_right_update = -max_speed_threshold

            # NOTE: if face is to the right of the drone, the distance will be negative, but
            # the drone has to have positive power so I am flipping the sign
            left_right_update = left_right_update * -1

            if up_down_update > max_speed_threshold:
                up_down_update = max_speed_threshold
            elif up_down_update < -max_speed_threshold:
                up_down_update = -max_speed_threshold
                
                
            if forward_update > max_speed_threshold:
                forward_update = max_speed_threshold
            elif forward_update < -max_speed_threshold:
                forward_update = -max_speed_threshold

            #print(int(left_right_update), int(up_down_update), int(forward_update))
            #int(forward_update // 2)

            me.send_rc_control(0, int(forward_update // 2), int(up_down_update // 2), int(left_right_update // 2))
            
            #Write video:
            if capture :
                out.write(frame)
        
    return(frame)


# vid = cv2.VideoCapture(0)
# while True :
#     ret, frame = vid.read()
#     me = False
    
#     # Function for face detection:
#     detect_track_face(me, frame, track=False)
#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# out.release()
# cv2.destroyAllWindows()


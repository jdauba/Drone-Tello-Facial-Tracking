# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:24:13 2021

@author: jerome
"""
import mtcnn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import tello
from time import sleep
import sys, os
import collections
sys.path.append('C:/Users/jerom/Desktop/Projet Tello/')
from objcenter import ObjCenter
from pid import PID

#Init face center
#face_center = ObjCenter(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_center = ObjCenter(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
detector = mtcnn.MTCNN()


#Init PID :
left_right_pid = PID(kP=0.7, kI=0.0001, kD=0.05)
up_down_pid = PID(kP=0.8, kI=0.0001, kD=0.1)
left_right_pid.initialize()
up_down_pid.initialize()

#Forward PID :
## Agressive mode :
wref = 90
#kp=1.8
forward_pid = PID(kP=1.7, kI=0.0001, kD=0.12)
forward_pid.initialize()
K_forward = 1.8
K_backward = 0.5
K_up = 3

## Soft mode :
# wref = 75
# #kp=1.8
# forward_pid = PID(kP=1.6, kI=0.0001, kD=0.1)
# forward_pid.initialize()
# K_forward = 1.6
# K_backward = 0.4
# Max speed :
max_speed_threshold = 120
search_speed = 70
#Init:
left_right_error = 0
count = 0
#Face recognition
id_face = 0
names = ['Searching...', 'Jerome Dauba', 'Jean-Charles Levy']
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
#♦ Frame buffer for face detection:
most_common_face = [0,0,0,0,0,0,0,0,0,0,0,0,0]

#Capture video :
capture = False

# TARGET NAME TO FOLLOW :
target = 1

if capture :
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Videos/Drone16.avi',fourcc, 20.0, (720,480))
    
def search_face(me, search):
    global count
    # Searching function when no faces are detected
    if count < 30:
        me.send_rc_control(0, 0, 0, 0)
        count += 1
    elif left_right_error > 0 :
        me.send_rc_control(0, 0, 0, -search_speed)
    elif left_right_error < 0:
        me.send_rc_control(0, 0, 0, search_speed)
    


def detect_track_face(me, frame, track):
    global left_right_error, count
    target = 0
    
    H,W,_ = frame.shape
    img_tr = cv2.flip(frame, -1) # Flip vertically
    gray = cv2.cvtColor(img_tr,cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray) #Ne marche pas : pq ?
    # On crée un "centre de frame" légèrement décalé vers le haut de l'image
    centerX = W//2
    centerY = H//3
    frame_center = (centerX, centerY)
    
    #Draw a red circle at the center of a frame:
    cv2.circle(frame, center=(centerX,centerY), radius=5, color=(0,0,255), thickness=-1)
    
    # Drawing rectangle for detected face :
    # objectLoc = face_center.update(frame, frameCenter=None)
    objectLoc = detector.detect_faces(frame)
    # ((objX,objY), rect, d) = objectLoc
    
    # if objX is None or objY is None :
    #     objX, objY = centerX, centerY

    try:
        rect = objectLoc[0]['box']
        x, y, w, h = rect
        objX, objY = x+w/2, y+h/2
        print('ObjX', objX)
        print('ObjY', objY)
        #print("WIDTH :", w)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        face_detected = True
        
        #Face recognition:
        id_face, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        if confidence < 170 :
            id_face = 0
            
        most_common_face.pop(0)
        most_common_face.append(id_face)
        print(most_common_face)
            
        id_face = collections.Counter(most_common_face).most_common(1)[0][0]
            
        name_face = names[id_face]
        
        cv2.arrowedLine(frame, frame_center, (objX, objY), color=(0,255,0), thickness=2)
        
    except:
        print("Aucun visage détecté")
        face_detected = False
        id_face =0
        #objX, objY = centerX, centerY
        pass
    
    #Drawing an arrow from center to face (represente the drone's displacement vector)

    
          
    if track :
                
        if not face_detected:
            # Drone search face
            print("Searching face...")
            search_face(me, search=True)

        else :
            count = 0
            left_right_error = centerX - objX
            left_right_update = left_right_pid.update(left_right_error, sleep=0)

            up_down_error = centerY - objY
            up_down_update = up_down_pid.update(up_down_error, sleep=0)
            if up_down_error < 0:
                up_down_error *= K_up
            
            
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
            
            cv2.putText(frame, f"Target: {name_face},{confidence}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX,
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
            if target == 0:
                print('OK')
                #me.send_rc_control(0, int(forward_update // 2), int(up_down_update // 2), int(left_right_update //2))
            else :
                if id_face == target :
                    print('Good target found')
                    #me.send_rc_control(0, int(forward_update // 2), int(up_down_update // 2), int(left_right_update // 2))
                else :
                    print('No target found, going stationnary')
                    #me.send_rc_control(0, 0, 0, 0)
            
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


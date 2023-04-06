
# Libraries
import cv2
import mediapipe as mp
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import threading
from keras.models import load_model
from config import *

# Init variables
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

model = load_model('D:\doan1\models\model.h5')
ls_landmark = []
label = "None"
i=0
k=0
l=0
m=0

previous_label = "none"
# Create dataset of landmarks and timestamp
def make_landmark_timestamp(poseRet):
    ret = []
    for idx, lm in enumerate(poseRet.pose_landmarks.landmark):
        ret.append(lm.x)
        ret.append(lm.y)
        ret.append(lm.z)
        ret.append(lm.visibility)
    return ret

# Draw landmarks on image
def draw_landmark(mpDraw, poseRet, frame):
    mpDraw.draw_landmarks(frame, poseRet.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return frame
# Interface with user

def draw_label(label, frame):
    global i,k,l,m
    global  previous_label
    
    if(label =="ngoi" and label != previous_label):
        i = i+1
    if(label == "dung" and label != previous_label):
        k = k+1
    
    if(label == "2tay" and label != previous_label):
        l = l+1
    
    if(label == "1tay" and label != previous_label):
        m = m+1
    
   
    previous_label = label
    
    
    text = "Class: {}".format(label)
    text1 = "So lan ngoi {}".format (i)
    text2 = "So lan dung {}".format (k)
    text3 = "So lan 2tay {}".format (l)
    text4 = "So lan 1tay {}".format (m)
    pos = (10,30)
    scale = 1
    thickness = 2
    lineType = 2
    fontColor = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                text,
                pos,
                font,
                scale,
                fontColor,
                thickness,
                lineType)
    lineType2 = 2
    pos1 = (10,60)
    cv2.putText(frame,
                text1,
                pos1,
                font,
                scale,
                fontColor,
                thickness,
                lineType2)
    pos1 = (10,90)
    cv2.putText(frame,
                text2,
                pos1,
                font,
                scale,
                fontColor,
                thickness,
                lineType2) 
    pos1 = (10,120)
    cv2.putText(frame,
                text3,
                pos1,
                font,
                scale,
                fontColor,
                thickness,
                lineType2)
    pos1 = (10,150)
    cv2.putText(frame,
                text4,
                pos1,
                font,
                scale,
                fontColor,
                thickness,
                lineType2)
    return frame

def detect(model, ls_landmark):
    global label
    tensor = np.expand_dims(ls_landmark,axis=0)
    result = model.predict(tensor)
    label = classes[np.argmax(result[0])]
    print(np.round(np.array(result[0]),3))
    

# Extract classes
files = os.listdir('D:/doan1/data')
classes = []
for path in files:
    classes.append(path.split('.')[0])
list.sort(classes)

while True:
    ret, frame = cap.read()
    if (ret):
        # Show input
        cv2.imshow('camera', frame)
        if cv2.waitKey(1)==ord('q'):
            break
        
        # Convert to RGB and create pose estimation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poseRet = pose.process(rgb)

        # Draw and create data
        if (poseRet.pose_landmarks):
            landmark = make_landmark_timestamp(poseRet)
            ls_landmark.append(landmark)
            frame = draw_landmark(mpDraw, poseRet, frame)

        # Inference
        if (len(ls_landmark)==N_TIME):
            t = threading.Thread(
                target = detect,
                args = (model, ls_landmark)
            )
            t.start()
            ls_landmark = []

        # Draw frame count
        frame = draw_label(label, frame)
        
        # Show pose
        cv2.imshow('pose', frame)
        

cap.release()
cv2.destroyAllWindows()
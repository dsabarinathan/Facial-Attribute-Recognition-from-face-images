# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:53:59 2024

@author: SABARI
"""



import cv2
import math
import argparse
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()


#model = load_model("pre-trained_weights/model_inception_facial_keypoints.h5",custom_objects={"Adamw":tfa.optimizers.AdamW},compile=False) # updated the loading function
faceNet = load_model("pre-trained_weights-20230405T090303Z-001/pre-trained_weights/model_face_net_files/",custom_objects={"Adamw":tfa.optimizers.AdamW})

facialList = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
      'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
      'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
      'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
      'Heavy_Makeup', 'High_Cheekbones', 'Male','Mouth_Slightly_Open',
      'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
      'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
      'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
      'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
      'Wearing_Necktie', 'Young']

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
image_batch = np.zeros((1,128,128,3))

while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = detector(frame)
    if not dets:
        print("No face detected")
        continue
    
    for det in dets:
        coord = [det.left(), det.top(), det.right(), det.bottom()]
        cropImage = frame[det.top(): det.bottom(),det.left():det.right()]
        
        
        image_batch[0] = cv2.resize(cropImage,(128,128))/256
        output = faceNet.predict(image_batch)
        

        resultImg = cv2.rectangle(rgb_image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)

        position0  = np.where(output[0]>0.5)[0]
        count = 25
        for i2 in range(len(position0)):
            if  "Eyeglasses" == facialList[position0[i2]]:
                print("Wearing EyeGlasses : Yes")
                cv2.putText(resultImg,"EyeGlasses :Yes", (det.left(),15+count),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            else:
                print(str(facialList[position0[i2]])+" "+str(np.round(output[0][position0[i2]],3)))
                cv2.putText(resultImg,str(facialList[position0[i2]]), (det.left(),15+count),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            count = count+15
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_RGB2BGR)

    cv2.imshow("Detecting facial attributes", resultImg)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

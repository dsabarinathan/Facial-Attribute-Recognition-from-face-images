
import dlib
import cv2
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from utils import draw_results

model = load_model("pre-trained_weights/model_inception_facial_keypoints.h5",custom_objects={"Adamw":tfa.optimizers.AdamW})



detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3

# Please use the own video path for testing 

video_read = cv2.VideoCapture('/Dataset/glass_wearing_video_short.mp4')

length = int(video_read. get(cv2. CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
ret_val, img = video_read.read()
height,width,channel = img.shape

height = int(height/2)
width = int(width/2)

video_output = cv2.VideoWriter('/Dataset/output_video.mp4', fourcc, 8, (width,height))
    
image_batch = np.zeros((1,128,128,3))
for i in range(length-10):

    ret_val, img = video_read.read()
    resizedImage = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
    rgb_image = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        coord = [det.left(), det.top(), det.right(), det.bottom()]
        cropImage = resizedImage[det.top(): det.bottom(),det.left():det.right()]
        image_batch[0] = cv2.resize(cropImage,(128,128))/256

        output = model.predict(image_batch)
        rgb_image = draw_results(rgb_image,det,output)
        cv2.rectangle(rgb_image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
  
    rgb_image1 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
    video_output.write(rgb_image1)

video_output.release()
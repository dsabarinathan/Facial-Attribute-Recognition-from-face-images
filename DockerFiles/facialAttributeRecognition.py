import cv2
import math
import argparse
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from flask import Flask, request, jsonify
import cv2
import json
import numpy as np
import dlib 

app = Flask(__name__)


class faceNet():

    def __init__(self,path="/app/models/model_inception_facial_keypoints.h5"):

        self.model = load_model("/app/models/model_inception_facial_keypoints.h5",custom_objects={"Adamw":tfa.optimizers.AdamW},compile=False) # updated the loading function

        self.facialList = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
            'Heavy_Makeup', 'High_Cheekbones', 'Male','Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
            'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young']
        self.detector = dlib.get_frontal_face_detector()

    def run(self,image,thresh=0.5):
        #if len(image.shape) == 3:
        #   image1 = np.expand_dims(image, axis=0)
        
        image_batch = np.zeros((1,128,128,3))

        
        dets = self.detector(image)
        face_results = []
        for det in dets:
            faceTemp ={}
            results =[]
            coord = [det.left(), det.top(), det.right(), det.bottom()]
            cropImage = image[det.top(): det.bottom(),det.left():det.right()]
            image_batch[0] = cv2.resize(cropImage,(128,128))/256

            # predict the facial landmarks
            output = self.model.predict(image_batch)
            
            position0  = np.where(output[0]>thresh)[0]
            

            for pos in range(len(position0)):
                temp = {}
                temp["label"] = self.facialList[position0[pos]]
                temp["prob"] = str(output[0][position0[pos]])

                results.append(temp)
            faceTemp["face"] = results
            faceTemp["coord"] = coord
            face_results.append(faceTemp)


        return face_results
    

@app.route('/',methods=["POST"])
def index():

    image = request.files['file'].read()
    npimg = np.fromstring(image, np.uint8)
    image_np = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    
    #resizedimage = cv2.resize(image_np,(128,128))/256
    results = faceAPP.run(image_np)

    result_json={}
    result_json["result"] = results
    return jsonify(result_json)

if __name__ == '__main__':
    faceAPP = faceNet()
    app.run(debug=True, host='0.0.0.0')
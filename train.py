import os
from sklearn.model_selection  import train_test_split
import numpy as np
import argparse
from model_face import faceNet
from model_vit import vitNet

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='facial-attribute-extraction')
    parser.add_argument("--imagepath", type=str,dest="data_path" ,help="Path of image ",default='./data/imageFile100000.npz',action="store")
    parser.add_argument("--labelpath", type=str,dest="label_path" ,help="Path of image label",default='./data/labelFile100000.npz',action="store")
    parser.add_argument("--model", type=str,dest="model_type" ,help="Type of model used to train",default='facenet',action="store")
    
    args = parser.parse_args()
      
    assert args.data_path[-3:]=="npz","The training file format should be npz. Please replace the training file"
    assert args.label_path[-3:]=="npz","The label file format should be npz. Please replace the training file"
    
    
    ## loading the preprocessed CelebFaces Attributes dataset
    
    data_x = np.load( args.data_path)
    data_y = np.load(args.label_path)
    data_x = data_x['image_arr']
    data_y = data_y['label_arr']
    
    # replace the -1 one value with zero for training the model.
    data_y[data_y==-1] = 0
    
    # because of memory constrain , i am using 50000 sample for training the model. 
    data_x=data_x[0:50000]
    data_y=data_y[0:50000]
    
    
    # split the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    
    del data_x,data_y # remove the original dataset
    
    # image normalization 
    
    x_train = np.float32(x_train/255)
    
    x_test = np.float32(x_test/255)
    
    labels = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young']
    
    
    # the preprocessed dataset image size is 128x128x3
    if args.model_type=="facenet":
        print("facenet model training started...")

        model = faceNet(img_width=128,img_height=128) 
        model.build()
        model.run(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,validation_split=0.1)
    elif args.model_type=="vit":
        print("vit classifier model training started...")

        model = vitNet() 
        model.create_vit_classifier(input_shape = (128, 128, 3),num_classes = 40)
        model.run(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,validation_split=0.1)
        
        
    print("Training has been completed")
    
    
    
    
    
    
    



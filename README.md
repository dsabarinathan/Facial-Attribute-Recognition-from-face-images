
# Facial Attribute Recognition from face images

This is a Keras implementation of a CNN for facial attribute recognition. I trained Visual Transformer and facenet for facial attribute extraction. 

# Dependencies
- Python3.6+

# Tested on

- Ubuntu 16.04, Python 3.6.9, Tensorflow 2.3.0, CUDA 10.01, cuDNN 7.6


# Dataset
I trained the face attribute extraction models with [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

You can download the preprocessed dataset from the below link. I cropped the faces and converted them into RGB format. 
The dataset contains 100000 images with facial attributes. 
https://drive.google.com/drive/folders/1iffYL-rB-3MbqI-TfFFHU6Wc-JaYHgGz?usp=sharing

# Train

```

python train.py --imagepath=/data/imageFile100000.npz --labelpath=/data/labelFile100000.npz

```

# Testing

```

python demo.py

```

# Testing in real time using the webcamera

```

python realtime_testing.py

```

# Pre_trained weights

Please use the below weights for testing.
https://drive.google.com/drive/folders/1NWz2E3b75mO_Ox8tb9d77vBi8dNHUv1T?usp=sharing


# Model results:


| Model | Train Accuracy  |  Validation Accuracy  |  Test Accuracy  |
| :------: | :------: | :-------: | :-------: |  
| VIT  |  81.2| 82  | 81.28  |
| FaceNet  | 84.5  | 85.71  |86.25|
|[InclusiveFaceNet](https://arxiv.org/pdf/1712.00193.pdf)| | |90.96 |

# Validation Dataset results

![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/validation_output_1.png)


# Test sample 

I used the bigbangtheory cast image as a testing image. Please find the person's result.

![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/testImage.jpg)

![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/sample_output_1.png)


# Output Video
![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/output_video_1.gif)

# Dcoker Installation Steps: 


# References:
  [FaceNet Tensorflow](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)

  [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
  
  [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)


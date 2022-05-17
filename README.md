
# Facial Attribute Recognition from face images

This is a Keras implementation of a CNN for facial attribute recognition. I trained Visual Transformer and facenet for facial attribute extraction. 

# Dependencies
- Python3.6+

# Tested on:

- Ubuntu 16.04, Python 3.6.9, Tensorflow 2.3.0, CUDA 10.01, cuDNN 7.6


# Dataset:
I trained the face attribute extraction models with [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

You can download the preprocessed dataset from the below link. I cropped the faces and converted them into RGB format. 
The dataset contains 100000 images with facial attributes. 
https://drive.google.com/drive/folders/1iffYL-rB-3MbqI-TfFFHU6Wc-JaYHgGz?usp=sharing

# Model results:


| Model | Train Accuracy  |  Validation Accuracy  |  Test Accuracy  |
| :------: | :------: | :-------: | :-------: |  
| VIT  |  81.2| 82  | 81.28  |
| FaceNet  | 84.5  | 85.71  |86.25|
|[InclusiveFaceNet](https://arxiv.org/pdf/1712.00193.pdf)| | |90.96 |

# Validation Dataset results

# Test sample 

I used the bigbangtheory cast image as a testing image. Please find the person's result.

![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/testImage.jpg)

![alt text](https://github.com/sabaridsn/Facial-Attribute-Recognition-from-face-images/blob/main/sample_output.png)




# Facial Attribute Recognition from face images

[![License: GPL](https://img.shields.io/badge/License-GPL-yellow.svg)](https://opensource.org/licenses/GPL-3.0) ![test](https://img.shields.io/static/v1?label=By&message=Tensorflow&color=red)

This is a Keras implementation of a CNN for facial attribute recognition. I trained Visual Transformer and facenet for facial attribute extraction. 

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dsabarinathan/Facial-Attribute-Recognition-from-face-images&type=Date)](https://star-history.com/#dsabarinathan/Facial-Attribute-Recognition-from-face-images&Date)

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

# Docker Installation Steps: 

- Step 1: Download the Pretrained facenet model and create the new folder inside the DockerFiles place it. i.e /DockerFiles/models/

```
/DockerFiles/models/model_inception_facial_keypoints.h5
```
- Step 2: Use the below command for docker compose.

```
docker-compose up -d
```
- Step 3: Run the following the command to build the docker image.

```
 docker build -t facial_attribute .

```
- Step 4: Start the detection service. 

```
 docker run -it facial_attribute
```

- Step 5: Pass the image for testing. 

```
curl -X POST -F 'file=@/home/couger/Desktop/1.jpg' http://172.17.0.2:5000/

```


- Step 6: JSON results format:


```
{
  "result": [
    {
      "coord": [
        607,
        65,
        711,
        169
      ],
      "face": [
        {
          "label": "Attractive",
          "prob": "0.5497437"
        },
        {
          "label": "Male",
          "prob": "0.8896191"
        },
        {
          "label": "No_Beard",
          "prob": "0.92911637"
        },
        {
          "label": "Young",
          "prob": "0.92061347"
        }
      ]
    },
    {
      "coord": [
        1149,
        131,
        1235,
        218
      ],
      "face": [
        {
          "label": "Big_Nose",
          "prob": "0.5611748"
        },
        {
          "label": "Male",
          "prob": "0.96252704"
        },
        {
          "label": "Mouth_Slightly_Open",
          "prob": "0.78494644"
        },
        {
          "label": "No_Beard",
          "prob": "0.5100374"
        },
        {
          "label": "Smiling",
          "prob": "0.7040582"
        },
        {
          "label": "Young",
          "prob": "0.8379371"
        }
      ]
    },
    {
      "coord": [
        803,
        150,
        889,
        237
      ],
      "face": [
        {
          "label": "Attractive",
          "prob": "0.59744525"
        },
        {
          "label": "Heavy_Makeup",
          "prob": "0.552807"
        },
        {
          "label": "No_Beard",
          "prob": "0.986242"
        },
        {
          "label": "Wearing_Lipstick",
          "prob": "0.692116"
        },
        {
          "label": "Young",
          "prob": "0.93902016"
        }
      ]
    },
    {
      "coord": [
        976,
        141,
        1062,
        227
      ],
      "face": [
        {
          "label": "Eyeglasses",
          "prob": "0.6799438"
        },
        {
          "label": "Male",
          "prob": "0.7749488"
        },
        {
          "label": "No_Beard",
          "prob": "0.9461371"
        },
        {
          "label": "Young",
          "prob": "0.6490406"
        }
      ]
    },
    {
      "coord": [
        179,
        150,
        266,
        237
      ],
      "face": [
        {
          "label": "Male",
          "prob": "0.9068607"
        },
        {
          "label": "Mouth_Slightly_Open",
          "prob": "0.91096807"
        },
        {
          "label": "No_Beard",
          "prob": "0.623013"
        },
        {
          "label": "Smiling",
          "prob": "0.8010901"
        },
        {
          "label": "Wearing_Hat",
          "prob": "0.57096326"
        },
        {
          "label": "Young",
          "prob": "0.88812125"
        }
      ]
    },
    {
      "coord": [
        446,
        158,
        549,
        261
      ],
      "face": [
        {
          "label": "Big_Nose",
          "prob": "0.7039994"
        },
        {
          "label": "Eyeglasses",
          "prob": "0.87806904"
        },
        {
          "label": "High_Cheekbones",
          "prob": "0.596"
        },
        {
          "label": "Male",
          "prob": "0.9493711"
        },
        {
          "label": "Mouth_Slightly_Open",
          "prob": "0.82170117"
        },
        {
          "label": "No_Beard",
          "prob": "0.86256987"
        },
        {
          "label": "Smiling",
          "prob": "0.88448894"
        }
      ]
    },
    {
      "coord": [
        304,
        170,
        390,
        256
      ],
      "face": [
        {
          "label": "Bangs",
          "prob": "0.56573707"
        },
        {
          "label": "Eyeglasses",
          "prob": "0.65550697"
        },
        {
          "label": "High_Cheekbones",
          "prob": "0.67516124"
        },
        {
          "label": "Mouth_Slightly_Open",
          "prob": "0.8242004"
        },
        {
          "label": "No_Beard",
          "prob": "0.9694848"
        },
        {
          "label": "Smiling",
          "prob": "0.8470793"
        },
        {
          "label": "Young",
          "prob": "0.6668907"
        }
      ]
    }
  ]
}

```

# References:
  [FaceNet Tensorflow](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)

  [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
  
  [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)


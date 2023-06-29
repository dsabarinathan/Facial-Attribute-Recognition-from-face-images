FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

# Install OpenCV package
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install OpenCV
RUN pip install opencv-python-headless
#Install Libraries
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install flask
RUN pip install requests
RUN pip install numpy
RUN pip install tensorflow-addons
RUN pip install dlib

COPY ./models/model_inception_facial_keypoints.h5 /app/models/model_inception_facial_keypoints.h5

COPY facialAttributeRecognition.py /app/facialAttributeRecognition.py

CMD ["python", "/app/facialAttributeRecognition.py"]


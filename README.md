# my-FaceNet-tf
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg?style=plastic)
![TensorFlow 0.12](https://img.shields.io/badge/tensorflow-0.12-green.svg?style=plastic)
![cuDNN 6.0](https://img.shields.io/badge/cudnn-6.0-green.svg?style=plastic)

A face recognition pipeline with [FaceNet model](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

This is my own FaceNet implementation.

### Requirements

* Python 3.5
* OpenCv 3.11
* Dlib 19.4
* Tensorflow 0.12
* Cudnn 6.0
* Scikit-learn 0.19.1
* Pillow 4.2.1

### Step 1: 
Face detection with OpenCV haar, dlib or MTCNN detector

### Step 2: 
Face alignment with OpenCV haar aligner, Dlib facial landmarks or MTCNN aligner

### Step 3: 
Feature Extractor ( optinal) with PCA or FaceNet embeddings

### Step 4: 
Face classification with FaceNet or Scikit-learn SVM classifiers

### Datasets: 
MS-Celeb-1M, LFW

### Results:

![IMAGE_DESCRIPTION](https://github.com/CerenGuzelTurhan/my-FaceNet-tf/blob/main/model_res.PNG)

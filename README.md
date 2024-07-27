# Emotion Detection

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.3.0-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-brightgreen.svg)
![Gradio](https://img.shields.io/badge/Gradio-API-brightgreen.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.18.1-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.0.1-yellow.svg)
![PIL](https://img.shields.io/badge/PIL-Image-brightgreen.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.10.0-blue.svg)

A Deep learning project for detecting emotions from facial expressions using Convolutional Neural Networks (CNN), Residual Neural networks (ResNet) and Transfer Learning. The project uses the FER-2013 dataset from Kaggle, leverages TensorFlow and Keras for building the models, and utilizes various image processing libraries and techniques.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Challenges](#challenges)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)

## Introduction
This project aims to detect emotions from facial expressions. The emotions detected include angry, disgust, fear, happy, neutral, sad, and surprise. The dataset used for this project is the FER-2013 dataset from Kaggle. The project explores the use of custom CNN models, including ResNet50V2 and VGG16, to achieve the best performance.

## Features
- **Emotion Detection:** Classifies facial expressions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise.
- **Data Augmentation:** Uses techniques to handle class imbalance and improve model robustness.
- **Model Analysis:** Evaluates models using precision, accuracy, recall, and F1 score.
- **Model Deployment:** Deploys the best model using Gradio API and OpenCV for real-time emotion detection.

## Challenges
- **Class Imbalance:** The dataset has fewer samples for specific classes like disgust. This was tackled using image augmentation techniques to generalize better and improve model robustness.
- **Image Class Weights:** Addressing class imbalance by assigning weights to different classes during model training.
- **Data Scarcity:** Limited image samples for certain classes were augmented to increase the diversity of training data.

## Dataset
The dataset used in this project is the FER-2013 dataset from Kaggle. It contains 35,887 grayscale images of faces, each labeled with one of seven emotion categories.

### Dataset Source
You can download the dataset from [Kaggle's FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

### Data Preprocessing
The images are resized to a consistent shape, and data augmentation techniques are applied to handle class imbalance and improve the model's generalization capabilities. Techniques such as rotation, shifting, and flipping are used to create a more robust training set.

## Model Architecture
The project explores several CNN architectures, including:

- **Custom CNN Model:** Designed and built from scratch to detect emotions.
- **ResNet50V2:** A pre-trained model fine-tuned for emotion detection.
- **VGG16:** Another pre-trained model used for comparison.

### Transfer Learning
The final and best model was achieved using transfer learning with ResNet50V2, which provided a balance between performance and computational efficiency.

### Performance Metrics
- **Precision**
- **Accuracy**
- **Recall**
- **F1 Score**

## Results
The project achieved an overall accuracy of 62% on the test set. The performance was evaluated using various metrics such as precision, recall, and F1 score. Detailed analysis and visualizations of the model's performance are provided in the Notebook.

## Deployment
The best-performing model was deployed using Gradio API and OpenCV, allowing for real-time emotion detection from images and video streams.



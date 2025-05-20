# multiimage_classification
A deep learning model that classifies images into multiple categories using a Convolutional Neural Network (CNN). Trained on a labelled image dataset for high accuracy.

🖼️ Multi-Class Image Classification Using CNN
This project implements a Convolutional Neural Network (CNN) to classify images into multiple categories. It demonstrates the use of deep learning for multi-class image recognition tasks using Keras and TensorFlow.

🎯 Objective
To train a robust CNN model that can accurately classify images across multiple categories using supervised learning and image augmentation techniques.

🧠 Technologies Used
Python 3.x

TensorFlow / Keras – for model creation and training

OpenCV / PIL – for image loading and preprocessing

NumPy & Pandas – for data handling

Matplotlib / Seaborn – for visualizations

📂 Notebooks Overview
✅ multipleimage_classification_training.ipynb
Preprocessing of training and validation datasets using ImageDataGenerator

CNN architecture design (multiple convolutional and pooling layers)

Model compilation and training

Accuracy/loss visualization

Model saved for later inference

✅ multipleimage_classification_testing.ipynb
Loads the trained model

Processes new/test images

Predicts class labels

Visualises predictions and prints confidence scores

🗃️ Dataset Structure
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
├── test/
│   ├── class1/
│   ├── class2/
│   └── class3/
Images are stored in subfolders by class name.

Training and testing sets are separated.

🧪 Model Architecture
2–3 Convolutional layers with ReLU and MaxPooling

Flattening + Fully Connected Dense layers

Dropout for regularization

Output layer with Softmax for multi-class classification

📊 Model Performance
Training and validation accuracy/loss plotted over epochs

Model achieves high accuracy on validation data (based on notebook results)
Final model is saved using .h5 format

🌟 Features
Multi-class prediction with softmax

Easy-to-train and test

Supports new image predictions

Clear and reusable notebook code

📬 Contact Created by Nireeha Padala 
📧 [nireehap@gmail.com] 
🔗 [www.linkedin.com/in/nireeha-padala-6a71ab2a0] | [https://github.com/Nireehapadala]

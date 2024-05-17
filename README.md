# Obesity Prediction Model

This project implements a machine learning model to predict obesity based on various demographic and lifestyle factors. The model is trained using a RandomForestClassifier and can be used to make predictions on new data.
## Table of Contents

- [Introduction](#Introduction)
- [Installation](#Installation)
- [Usage](#Usage)
- [Dependencies](#Dependencies)

## Introduction
Obesity is a significant public health issue affecting individuals worldwide. This project aims to develop a predictive model that can classify individuals into different obesity categories based on their age, gender, height, weight, and other relevant features.

The model is trained on a dataset containing information about individuals' characteristics and their obesity status. After training, the model can be used to predict obesity levels for new individuals based on their provided attributes.

## Installation
To use this project, follow these steps:
1. Clone the repository to your local machine: `git clone https://github.com/SofiaMelashchuk/Obesity`
2. Install the required Python packages using pip: `pip install -r requirements.txt`

## Usage
Training the Model
To train the obesity prediction model, run the `train.py` script.
This script will read the dataset, preprocess the data, train the RandomForestClassifier model, and save the trained model along with feature importance information.

Making Predictions
To make predictions using the trained model, prepare a CSV file (new_data_for_prediction.csv) containing new data for prediction. Then, run the `predict.py` script. 
This script will preprocess the input data, load the trained model, perform predictions, and save the prediction results to a CSV file (prediction_result.csv).

## Dependencies
- pandas
- scikit-learn
- joblib

# Credit Card Fraud Detection using Autoencoders in Keras

This project implements an unsupervised deep learning approach to detect fraudulent credit card transactions using Autoencoders in Keras with a TensorFlow backend.

## 🔍 Overview

Credit card fraud detection is a classic anomaly detection problem, where the goal is to identify rare events (fraudulent transactions) from a highly imbalanced dataset. In this project, we use autoencoders trained only on normal (non-fraudulent) transactions to learn their patterns. Fraudulent transactions are detected as outliers based on reconstruction error.

## 🚀 Features

- Unsupervised anomaly detection using deep autoencoders
- Built with Keras and TensorFlow
- Trained on normal transactions only
- Detects fraud based on reconstruction error threshold
- Evaluation using precision, recall, F1-score, ROC curve, and confusion matrix
- Clean and reproducible code

## 🧠 Model Architecture

- Input layer: Normalized features of transactions
- Encoder: Dense layers that compress the input
- Decoder: Dense layers that reconstruct the input
- Loss: Mean squared error (MSE) between original and reconstructed input

## 📊 Dataset

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in September 2013
- 284,807 transactions with only 492 frauds (~0.17%)



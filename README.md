# Breast Cancer Classification using Machine Learning

## 1. Objective
Build and compare five machine learning classification models to predict whether a breast tumour is malignant or benign. An interactive Streamlit application allows users to upload test data and view predictions and evaluation metrics.

## 2. Dataset
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Records: 569
- Features: 30
- Target: Diagnosis (0 = Malignant, 1 = Benign)
- Training records: 455
- Testing records: 114

## 3. Models and Results

| Model | Accuracy | AUC |
|---|---:|---:|
| Logistic Regression | 98.25% | 0.9954 |
| Decision Tree | 91.23% | 0.9157 |
| KNN | 95.61% | 0.9788 |
| Naive Bayes | 93.86% | 0.9878 |
| SVM | 98.25% | 0.9950 |

Other evaluation metrics include Precision, Recall, F1-score and MCC.

## 4. Observations
- Logistic Regression achieved the highest AUC and is the overall best model.
- SVM achieved the same accuracy as Logistic Regression.
- KNN and Naive Bayes also performed well.
- Decision Tree achieved the lowest accuracy among the five models.

## 5. Streamlit Application
The application allows users to upload a test CSV, select a trained model, view predictions, accuracy and other evaluation metrics, confusion matrix and classification report. Models are loaded from saved files; no retraining occurs during upload.

Run using:
pip install -r requirements.txt
streamlit run app.py

## 6. Project Files
- app.py: Streamlit application
- requirements.txt: Python dependencies
- test_data.csv: Test dataset
- model_comparison.csv: Model evaluation results
- model/: Five saved trained models

## 7. Project Links
GitHub Repository: To be added
Live Application: To be added

## 8. Disclaimer
This project is for academic purposes only and is not intended for medical diagnosis.

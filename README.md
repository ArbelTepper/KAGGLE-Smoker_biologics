# Smoking Prediction using Synthetic Health Data

This project uses a synthetic health dataset to predict whether an individual smokes based on various health-related features. The dataset includes a range of biometric data such as age, weight, cholesterol levels, and more, and is designed for training machine learning models to make predictions about smoking habits.

## Dataset

The dataset used in this project, [which can be found here](https://www.kaggle.com/competitions/playground-series-s3e24), consists of synthetic health records provided in CSV format:

- **train.csv**: The training dataset containing features and labels (smoking status).
- **test.csv**: The test dataset for making predictions.
- **sample_submission.csv**: A sample submission file for Kaggle competitions.

The columns in the dataset include:
- **age, height(cm), weight(kg), waist(cm), eyesight(left), eyesight(right), hearing(left), hearing(right), systolic, relaxation, fasting blood sugar, cholesterol, triglyceride, HDL, LDL, hemoglobin, urine protein, serum creatinine, AST, ALT, GTP, dental caries, smoking**

The target variable is the **smoking** column, where:
- `0`: Non-smoker
- `1`: Smoker

## Project Goal

The goal of this project is to build a machine learning model that predicts the probability of smoking based on the health metrics provided in the dataset.

I used a **HistGradientBoostingClassifier** as the model and focused on optimizing its performance using cross-validation and hyperparameter tuning.

## Approach

1. **Exploratory Data Analysis (EDA)**: 
    - Explored the dataset to identify missing values, data types, and the distribution of labels.
    - Identified and handled outliers by replacing them with boundary values.

2. **Data Preprocessing**:
    - No missing values were found in the dataset.
    - Features were selected and cleaned, focusing on continuous features for scaling.
    - Categorical features (like hearing and urine protein) were handled appropriately.

3. **Modeling**:
    - Built a pipeline with **HistGradientBoostingClassifier** and optimized it using **cross-validation**.
    - The model was tuned using **GridSearchCV** to find the best hyperparameters.

4. **Evaluation**:
    - The performance of the model was evaluated using **ROC AUC** as the metric, achieving an average score of **0.866** in cross-validation.

5. **Final Submission**:
    - The best model was trained on the entire training data and used to predict smoking probabilities for the test dataset.
    - My submission received a Kaggle competition score of 0.867, which is pretty good considering the best submission scored 0.879. 

## Conclusion

This project demonstrates the process of working with synthetic health data to build a predictive model for smoking habits. The dataset is well-structured, and using a tree-based model like HistGradientBoostingClassifier performed well despite the challenges of handling outliers and tuning the model.

## Future Work Ideas

- Experimenting with different classifiers and feature engineering techniques.
- Incorporating more advanced feature selection.
- Testing with different validation strategies or resampling techniques.

## Requirements

- `Python 3.x`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

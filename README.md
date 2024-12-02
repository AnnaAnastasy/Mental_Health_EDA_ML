# Mental Health Analysis and Prediction Project
## Project Overview
This project focuses on analyzing and predicting mental health conditions using a dataset containing various demographic, behavioral, and psychological attributes. The objective is to understand key factors associated with mental health and build a predictive model for identifying individuals at risk of depression.

## Dataset Details
- **Source**: [Dataset for Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e11).
- **Rows in Training Set**: 140,700
- **Columns**: 20
- **Target Variable**: Depression (binary: 1 indicates depression, 0 indicates no depression)
- **Features**:
  - **Demographic**: Age, Gender, City, Degree
  - **Behavioral**: Work/Study Hours, Sleep Duration, Dietary Habits
  - **Psychological**: Academic Pressure, Work Pressure, Job Satisfaction, Study Satisfaction
  - **Other**: Financial Stress, Family History of Mental Illness, Suicidal Thoughts

## Key Steps
- **Import Necessary Libraries**:
  - **Data Manipulation**: pandas, numpy
  - **Visualization**: matplotlib, seaborn, plotly
  - **Machine Learning**: XGBoost, CatBoost, scikit-learn
  - **Ensemble**: StackingClassifier, HistGradientBoostingClassifier

- **Data Cleaning**:
  - Addressed **missing values** in columns like Academic Pressure, CGPA, and Job Satisfaction.
  - Handled **categorical features** with a mix of encoding techniques.

- **Exploratory Data Analysis (EDA)**:
  - Visualized **distributions** of numerical and categorical variables.
  - Examined **relationships** between features and the target variable (Depression).
  - Highlighted **key trends**, such as the impact of work pressure and sleep duration on mental health.

- **Feature Engineering**:
  - Added **interaction terms** like Age_WorkPressure.
  - Applied **target encoding** to categorical features (e.g., City, Profession).

- **Model Building**:
  - Implemented three classifiers: **XGBoost**, **CatBoost**, and **HistGradientBoostingClassifier**.
  - Combined models using a **stacking ensemble** for better accuracy.
  - Conducted **hyperparameter tuning** for optimal performance.

- **Model Evaluation**:
  - **Cross-Validation Accuracy**: 94.34%
  - **Standard Deviation of Accuracy**: 0.12%

- **Prediction**:
  - Predictions on the test dataset were saved in `submission.csv`.

## Results
- Achieved a robust ensemble model with a **high accuracy of 94.34%** on cross-validation.
- Identified **key insights**:
  - Work pressure and academic pressure are significant contributors to mental health issues.
  - Sleep duration and dietary habits are important lifestyle factors.

## Requirements
To replicate the project, ensure the following Python libraries are installed:
```
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost catboost
```

## Acknowledgments
This project is a step toward leveraging data science for understanding and addressing mental health challenges. It is not a substitute for professional medical advice or diagnosis.


# Phishing Website Classification

A machine learning model to classify websites as either legitimate or phishing.

---

## Project Overview

This project cleans a dataset of website features, selects the most informative features using Mutual Information, and trains a Random Forest classifier to detect phishing sites. The focus is on achieving high recall for the phishing class to minimize the risk of missing actual threats.

## Workflow

1.  **Data Cleaning:** Loaded the `dataset_phishing.csv` file, removed duplicates, and handled non-informative columns.
2.  **Feature Selection:** Used Mutual Information to identify the top 12 most predictive features, as simple correlation was not effective.
3.  **Data Transformation:** Applied a log transformation (`log1p`) to highly skewed features to normalize their distributions and handle outliers.
4.  **Model Training:** Evaluated several models (Logistic Regression, SVM, Naive Bayes, Random Forest) using cross-validation. A Random Forest classifier was selected for its superior performance.
5.  **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal hyperparameters for the Random Forest model, optimizing for recall.
6.  **Evaluation:** Assessed the final model's performance on a test set using a comprehensive set of metrics and visualizations.

## Final Model Performance

The final model is a `RandomForestClassifier` with the following parameters:
- `n_estimators`: 100
- `max_depth`: 10
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `max_features`: 'sqrt'

### Key Metrics

| Metric                  | Score |
| ----------------------- | ----- |
| **Accuracy** | 0.94  |
| **F1-Score** | 0.94  |
| **ROC AUC** | 0.98  |
| **Precision-Recall AUC**| 0.98  |

### Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------- | ------ | -------- | ------- |
| **Benign (0)** | 0.94      | 0.94   | 0.94     | 4520    |
| **Phishing (1)** | 0.94      | 0.94   | 0.94     | 4484    |

### Confusion Matrix

|            | Predicted Benign | Predicted Phishing |
| ---------- | ---------------- | ------------------ |
| **Actual Benign** | 4263             | 257                |
| **Actual Phishing**| 286              | 4198               |

## Visualizations

The analysis includes several key visualizations to understand the data and model performance:
- **Feature Analysis:** KDE plots, Box plots, and Violin plots to show feature distributions per class.
- **Model Evaluation:**
    - ROC Curve
    - Precision-Recall Curve
    - Learning Curve
    - Validation Curve

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd your-repository-directory
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
3.  Run the script:
    ```bash
    python asd_project_minor.py
    ```

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

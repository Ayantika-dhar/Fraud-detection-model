#  Fraud Detection Using Random Forest (Financial Transactions)

This project implements a **machine learning model to detect fraudulent transactions** using a real-world financial dataset with over 6 million records. The model is built using a **Random Forest Classifier** with class imbalance handling and evaluated using industry-standard metrics.

---

##  Dataset Overview

- **Source**: Provided as part of an internship assignment
- **Rows**: 6,362,620
- **Columns**: 10
- **Target Variable**: `isFraud` (1 = fraud, 0 = not fraud)

---

##  Technologies Used

- Python 3.x
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn (`RandomForestClassifier`, metrics)
- Google Colab (for training)
- Joblib (for saving the model)

---

##  Workflow & Steps

### 1.  Data Preprocessing
- Loaded the dataset and performed EDA
- Encoded categorical feature `type` using `LabelEncoder`
- Removed identifier fields: `nameOrig`, `nameDest`
- Removed highly correlated features: `newbalanceOrig`, `newbalanceDest`
- Handled class imbalance using `class_weight='balanced'`

### 2.  Model Training
- Used `RandomForestClassifier` with:
  - `class_weight='balanced'`
  - `n_jobs=-1` for parallel processing
- Applied 70-30 **stratified train-test split**

### 3.  Evaluation Metrics
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)
- **ROC AUC Score**

---

##  Key Results

| Metric              | Value        |
|---------------------|--------------|
| Precision (Fraud)   | 94.08%       |
| Recall (Fraud)      | 69.68%       |
| ROC AUC Score       | 0.984        |

These results indicate that the model is highly effective at detecting fraudulent transactions with minimal false positives.

---


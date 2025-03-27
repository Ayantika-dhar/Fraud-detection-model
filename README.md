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


---

## 🚀 How to Run the Project

### Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Mount Google Drive (optional if you're loading data from Drive):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload your `Fraud.csv` dataset to Google Drive.
4. Open or paste the contents of `Fraud.py` into a new Colab notebook.
5. Update the file path in the script if needed:
   ```python
   df = pd.read_csv("/content/drive/MyDrive/Fraud.csv")
   ```
6. Run the notebook. The model will be trained and evaluated directly within Colab.

---

###  Running on Local Machine

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection.git
   cd fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `Fraud.csv` is placed in the project root directory.

4. Run the training script:
   ```bash
   python Fraud.py
   ```

5. Output:
   - Model training results, including confusion matrix, classification report, and ROC AUC score, will be printed in the terminal.

---



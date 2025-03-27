import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/Fraud.csv")  # Replace with your full path if needed

# Display basic info
print("Shape of dataset:", df.shape)
df.head()
print(df.columns)
df.isnull().sum()
df.duplicated().sum()
df.describe()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['amount', 'oldbalanceOrg', 'newbalanceOrig']])
plt.xticks(rotation=90)
plt.title("Boxplot for Numeric Features")
plt.show()
from sklearn.preprocessing import LabelEncoder
df['type'] = LabelEncoder().fit_transform(df['type'])
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
df.drop(['newbalanceOrig', 'newbalanceDest'], axis=1, inplace=True)

# Count fraud vs non-fraud (percentage)
fraud_counts = df['isFraud'].value_counts(normalize=True) * 100
print(fraud_counts)


# Step 1: Define features and target
X = df.drop(['isFraud'], axis=1)
y = df['isFraud']

# Step 2: Train-test split (stratified to maintain fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 3: Build Random Forest with class_weight
model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ROC AUC Score
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))



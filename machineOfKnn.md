
# 🧠 K-Nearest Neighbors (KNN) for Breast Cancer Classification


## 📌 What is KNN?

- **KNN** = K Nearest Neighbors
- **K**: Number of neighbors considered
- **Supervised Learning** algorithm
- **Lazy Learning** (no model is built in advance)
- **Non-parametric** (no assumptions about data distribution)
- **Non-linear**, **Non-deterministic**, and **Non-probabilistic**

---

## 📊 Example

Given **K = 5**, the algorithm picks the 5 nearest neighbors.  
If the majority belong to class **blue**, the prediction is **blue**.

---

## 📁 1. Data Loading and Cleaning

```python
import pandas as pd

# Load the dataset
df = pd.read_csv("breast-cancer-wisconsin.csv", sep=",")

# Clean column names
df.columns = df.columns.str.strip()

# Remove rows with missing values
df = df[df['Núcleos desnudos'] != '?'].reset_index(drop=True)

# Convert to integer
df['Núcleos desnudos'] = df['Núcleos desnudos'].astype(int)

# Drop 'Id' column if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Split features and labels
X = df.drop(columns=['Clase'])
y = df['Clase']
```

---

## ✂️ 2. Split Train/Test Sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

## ⚖️ 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Save column names
feature_columns = X_train.columns

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Back to DataFrame
X_train = pd.DataFrame(X_train, columns=feature_columns)
X_test = pd.DataFrame(X_test, columns=feature_columns)
```

---

## 🧪 4. Model Training & Evaluation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 👩‍⚕️ 5. Prediction for New Patients

```python
# Create new patient data
new_patients = {
    'Grosor del grumo': [4, 7, 1],
    'Uniformidad del tamaño de las células': [1, 4, 0],
    'Uniformidad de la forma de las células': [1, 4, 0],
    'Adhesión marginal': [1, 4, 0],
    'Tamaño de una sola célula epitelial': [2, 4, 0],
    'Núcleos desnudos': [1, 5, -3],
    'Cromatina suave': [3, 5, 1],
    'Núcleos normales': [1, 4, 0],
    'Mitosis': [1, 3, 0],
}
new_df = pd.DataFrame(new_patients)

# Scale the new data
new_df_scaled = scaler.transform(new_df)
new_df_scaled = pd.DataFrame(new_df_scaled, columns=feature_columns)

# Predict
predictions = knn.predict(new_df_scaled)
print("Predictions:", predictions)
```

---

## ✅ Final Summary

1. Loaded and cleaned the dataset.
2. Handled missing data and formatted columns.
3. Split into training and testing sets.
4. Scaled input features.
5. Trained a KNN classifier (k=5).
6. Evaluated on test data.
7. Made predictions on new synthetic patients.

---

📌 **Result Interpretation:**
- **0** = Benign tumor
- **1** = Malignant tumor

🧠 This project introduces KNN with a real-world medical dataset and demonstrates a complete ML pipeline from raw data to prediction.


# 🧠 Handwritten Digits Classifier (0–9)

## 📝 Project Description

In this project, we will build a **neural network** capable of classifying handwritten digits from **0 to 9** using the `sklearn.datasets.load_digits()` dataset.

This is a common image classification task and a great introduction to using neural networks for **computer vision**.

---

## 📦 1. Load Required Libraries

We will use TensorFlow/Keras for model building and training, and `matplotlib` for visualization.

```python
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
```

---

## 📥 2. Load the Digits Dataset

```python
digits = load_digits()
```

### Dataset Properties

```python
print("Keys of digits: \n", digits.keys())
```

```
dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
```

### Dataset shape

```python
digits.images.shape
```

```
(1797, 8, 8)
```

That means we have **1797 samples** of **8x8 pixel images** representing digits from 0 to 9.

---

## 🖼️ 3. Visualize a Sample Digit

Let's display the digit at index 900:

```python
plt.imshow(digits.images[900], cmap=plt.cm.gray_r)
plt.show()
```

Digit represented:

```python
digits.target[900]
```

```
4
```

This confirms that the label of the displayed image is the digit **4**.

---

## 🔀 4. Split the Data

We will divide the data into:

- **Training set**
- **Validation set**
- **Test set**

### Step 1: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.3, random_state=0)
```

Check dimensions:

```python
X_train.shape, X_test.shape, digits.images.shape, y_train.shape, y_test.shape, digits.target.shape
```

```
((1257, 8, 8), (540, 8, 8), (1797, 8, 8), (1257,), (540,), (1797,))
```

### Step 2: Train-Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
```

Check new shapes:

```python
X_train.shape, X_val.shape, y_train.shape, y_val.shape
```

```
((1068, 8, 8), (189, 8, 8), (1068,), (189,))
```

We now have:

- **Train**: 1068 samples
- **Validation**: 189 samples
- **Test**: 540 samples

---

## ✅ Status Summary

- Loaded the **digits dataset** from `sklearn.datasets`.
- Visualized the image at index 900 with label **4**.
- Split the data into training, validation, and test sets:
  - 60% training
  - 15% validation
  - 25% testing

---

## 🧮 Next Steps (not yet coded)

- Normalize and reshape the image data for use in neural networks.
- Convert labels to one-hot encoding with `to_categorical()`.
- Build and compile a simple neural network using `tf.keras.Sequential`.
- Train the model and evaluate accuracy on the validation and test sets.

---

## 📌 Extras

### Check proportions

```python
1257 * 0.15  # validation from train
# Output: 188.55

1797 * 0.3   # total test
# Output: 539.1
```
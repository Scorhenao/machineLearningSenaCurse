

# 🧠 Handwritten Digits Classifier (0–9)



## 📝 Project Description

In this project, we will build a **neural network** capable of classifying handwritten digits from **0 to 9** using the `sklearn.datasets.load_digits()` dataset.

This is a common image classification task and a great introduction to using neural networks for **computer vision**.



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


# Numpy

- is a library that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

- A problem is that u can't mix types of data, due to the fact that Numpy is a C library.

- The arrays in Numpy could be N-dimensional, which means that you can have arrays of arrays, and even matrices.

### Usage:

```py
pip install numpy

import numpy as np

#1 dimension array
array_1d = np.array([1, 2, 3])
print(array_1d)
# result is: [1 2 3]

array_1d.shape
# result is: (3,)


x = (1)
type(x) # result is: <class 'int'>

y = (1,)
type(y) # result is: <class 'tuple'>

#2 dimension array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(array_2d)
# result is: [[1 2 3]
#            [4 5 6]]

array_2d.dtype
# result is: dtype('int32') the int by default are of 64 bits

array_2d.shape
# result is: (2, 3) 2 rows and 3 columns

# 3 dimension array
array_3d = np.array(
    [
        [
            [1.2, 2.3, 3.4],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ]
    ]
    )
print(array_3d)
# result is: [[[ 1.2  2.3  3.4]
#             [ 4.2  5.3  6.4]]
#            [[ 7  8  9]
#             [10 11 12]]]

array_3d.dtype
# result is: dtype('float64')

array_3d.shape
# result is: (2, 2, 3) 2 arrays of 2 rows and 3 columns

array_3d.ndim
# result is: 3

#force the change of the data type
type(1 / 1)
# result is: 1.0 float(32 bits)

a = 1
b = 1 / 1

print(a == b) # result is: True because the data type is the same

print(a is b) # result is: False because the data type is the same but is in a different memory location

squared_array = np.sqrt(array_2d) #this print the square of the array
print(squared_array)
# result is: [[1.          1.41421356  1.73205081]
#            [2.          2.23606798  2.44948974]]

squared_array.dtype
# the data type is float(64 bits)

# change array 2d to float
array_2d_float = array_2d / 1
print(array_2d_float) # This i a CAST
print(array_2d_float.dtype) # result is: float64
# A cast is a conversion from one data type to another.

# list of the 15 first prime numbers
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# Slicing a list is getting a part of the list
print(primes[0:5]) # result is: [2, 3, 5, 7, 11]

```

# Pandas

- Pandas is a software library build on top of Numpy that provides high-performance, easy-to-use data structures and data analysis tools for Python. (data structures tabulated)

### Usage:

```py
pip install pandas

import pandas as pd

# DataFrame is a 2-dimensional table of data with columns of potentially different types. This alow tabulate functionalities like add, delete, update, rename, etc.

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': ['a', 'b', 'c']})
print(df)
# result is:
#       col1  col2  col3
# 0     1     4     a
# 1     2     5     b
# 2     3     6     c

# The head() method returns the first n rows of the DataFrame.

print(df.head())
# result is:
#       col1  col2  col3
# 0     1     4     a
# 1     2     5     b

# The tail() method returns the last n rows of the DataFrame.

print(df.tail())
# result is:
#       col1  col2  col3
# 1     2     5     b
# 2     3     6     c

# The sample() method returns a random sample of the DataFrame.

print(df.sample())
# result is:
#       col1  col2  col3
# 1     2     5     b

# The describe() method returns a summary of the DataFrame.

print(df.describe())
# result is:
#       col1  col2
# count  3.0  3.0
# mean   2.0  5.0
# std    1.0  1.0
# min    1.0  4.0
# 25%    1.5  4.5
# 50%    2.0  5.0
# 75%    2.5  5.5
# max    3.0  6.0
```

**Important:** if you change the data from excel to a csv file the separation of each coma will be with , besides a ; it is called data _sesgada_

#### organize data as csv

```py
import pandas as pd

df_salaries = pd.read_csv('salary_data.csv', sep=';')
df_salaries

# the file must be in the path of the folder

df_salaries.dtypes
# result is:
# ID int64
# income float64
# age int64
# gender object
# education_level float64

#Conditions filters
is_woman = df_salaries['gender'] == 'F'

df_womans = df_salaries.loc[is_woman]

df_womans

df_womans['income'].max()
# return the woman that gain the most money
```

```py
import pandas as pd

df.sample(4, random_state=1) # return 4 random rows from the dataframe with a fixed seed for reproducibility of the results

df.col1 # return the first column of the dataframe and the dtype

type(df.col1) # return the type of the first column of the dataframe

df.col1 is df['col1'] # return True if the first column of the dataframe is the same as the second column of the dataframe the df['col1'] is more readable

#slicing a dataframe
df[['col1', 'col2']] # return the first and second column of the dataframe

df['col3'].dtype # return a special type of panda like 'O' for object

df.dtypes # return the types of the columns of the dataframe
```

# Methods of pandas

```py
.head() # return the first n rows of the dataframe
.tail() # return the last n rows of the dataframe
.describe() # return a summary of the dataframe
.max() # return the maximum value of the dataframe
.min() # return the minimum value of the dataframe
.mean() # return the mean value of the dataframe
.median() # return the median value of the dataframe
.std() # return the standard deviation of the dataframe
.sample() # return a random sample of the dataframe
.dropna() # return the dataframe without the rows with missing values
```

# Filter information

```py
_the_toy_is_red = df['color'] == 'red'

df[_the_toy_is_red] # return the dataframe and put true if the color is red

df.loc[_the_toy_is_red] # return the dataframe where the color is red

df._the_toy_is_red.shape # return the shape of the dataframe

# divide only the reds in other dataframe

df_reds = df[_the_toy_is_red]

df_reds.shape # return the shape of the dataframe


# calculate the maximum salary for a person with level 1 of studies

df[df['education_level'] == 1]['income'].max() # return the maximum salary for a person with level 1 of studies

is_level_1 = df['education_level'] == 1 # return True if the education level is 1

is_level_1.value_counts() # return the number of people with level 1 of studies

df_level_1 = df[is_level_1] # return the dataframe where the education level is 1

max_salary = df_level_1['income'].max() # return the maximum salary for a person with level 1 of studies
print(f'The maximum salary for a person with level 1 of studies is {max_salary, 2}') # return the maximum salary for a person with level 1 of studies with 2 decimal places
```

# Group information

### Histograma

- a histogram is a graph that shows the frequency of values in a dataset

```py
df.groupby('education_level')['income'].mean() # return the mean of the income for each education level
df.show() # return the dataframe with the mean of the income for each education level and the number of people with each education level

# pd.read_csv("archivo.csv", sep=";")
# → read the data from a csv file.

# df.groupby('education_level')
# → group the data by education level.

# .plot(kind='hist', bins=20, alpha=0.5, legend=True)
# → generate a histogram of the mean income for each education level.:

# kind='hist' → specifies that the plot should be a histogram.

# bins=20 → divide the data into 20 bins intervals.

# alpha=0.5 → Add transparency to the histogram.

# legend=True → Add a legend for each level of education.

df.groupby('education_level')['income'].mean().plot(kind='hist', bins=20, alpha=0.5, legend=True)

```

#### Another solution

```py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. histograma of education level
ptl.hist(df['education_level'])
plt.show()

# 2. histograma of income for each level of education
condition = df['education_level'] == 0
df_filtered = df.loc[condition]
# plt.hist(df_filtered['income'])
df_filtered.hist(column='income')
# plt.show()

# another way
df.loc[df['education_level'] == 0].hist(column='income')
plt.show()
```

# graphics of boxs and whiskersa

- show in a unidimentional way the median, cuartiles and outliers, atipic values, etc.

```py
df.boxplot(column='income') # return the boxplot of the income column
ptl.show()
# empty circles in the graphic are the atipic dates
# the green line is the median

db.bloxplot(column='income', by='education_level') # return the boxplot of the income column by education level

```

# Primer KNN

- KNN = K nearest neighbors
- K = number of neighbors
- KNN is a supervised learning algorithm
- KNN is a lazy learning algorithm
- KNN is a non parametric algorithm
- KNN is a non linear algorithm
- KNN is a non deterministic algorithm
- KNN is a non probabilistic algorithm

#### example

- if k= 5
- kj = x1, x2, x3, x4, x5 nearest neighbors
- if there are more blue than red the prediction of the class blue
- x1, x2, x3, x4, x5 are in the class y1

```py
import pandas as pd
df = pd.read_csv('breast-cancer-wisconsin.csv', sep=',')

df.sample(5) # return 5 random rows of the dataframe

df['Núcleos desnudos'].value_counts() # return the number of rows for each value of the column 'Núcleos desnudos'

# what i'm suppose to do if there one class give me a ?
# it is because the type change from int to object
# kick values with ? in the column 'Núcleos desnudos'
df = df.loc[df['Núcleos desnudos'] != '?']

df =df.reset_index(drop=True) # reset the index of the dataframe after delete the ?

#it keeps like an object so we need to do a cast
df['Núcleos desnudos'] = df['Núcleos desnudos'].astype('int64') # change the type of the column 'Núcleos desnudos' to int64

#kick the column 'Id'
df = df.drop(columns=['Id'])
```

```py
# Vamos a dividir los datos como datos de entrenamiento (data training) y los datos de prueba (data testing)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape, X_test.shape # return the shape of the training data

#Corregido
# import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV y limpiar los nombres de las columnas
file_path = "breast-cancer-wisconsin.csv"
df = pd.read_csv(file_path, sep=',')
df.columns = df.columns.str.strip()  # Eliminar espacios en los nombres de las columnas

# Verificar valores únicos en la columna 'Núcleos desnudos'
if 'Núcleos desnudos' in df.columns and '?' in df['Núcleos desnudos'].values:
    df = df[df['Núcleos desnudos'] != '?']  # Eliminar filas con valores '?'

# Resetear el índice después de eliminar filas
df = df.reset_index(drop=True)

# Convertir la columna a tipo numérico si existe
if 'Núcleos desnudos' in df.columns:
    df['Núcleos desnudos'] = df['Núcleos desnudos'].astype(int)

# Eliminar la columna 'Id' si existe
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Separar en datos de entrada (X) y etiquetas de salida (y) si 'Clase' existe
if 'Clase' in df.columns:
    X = df.drop(columns=['Clase'])
    y = df['Clase']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Mostrar las dimensiones de los conjuntos de entrenamiento y prueba
    X_train.shape, X_test.shape

    # y is a sub serie of pandas

```

### Do a reescale of the caracteristics of $X_{train}$ and $X_{test}$

1. We'll save the list of originals columns in $X_{train}$
2. we'll reescale $X_{train}$ and $X_{test}$ with sklearn
3. we'll notice that $X_{train}$ and $X_{test}$ pass from dataframes of pandas to arrays of numpy
4. become $X_{train}$ and $X_{test}$ again dataframes of pandas

```py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#save columns
my_cols = x_train.columns

# reescale x_train and x_test
my_scaler = StandardScaler() # create the scaler
x_train = my_scaler.fit_transform(x_train) # apply the scaler
x_test = my_scaler.transform(x_test) # apply the scaler

# notice that x_train and x_test pass from dataframes to arrays and the quantity reprents the number of dimensions

# recover the dataframe format of pandas in x_train and x_test

x_train = pd.DataFrame(x_train, columns=my_cols)
x_test = pd.DataFrame(x_test, columns=my_cols)
```

# Usage of the first machine learning model knn and tain it with $X_{train}$ and $y_{train}$

```py
from sklearn.neighbors import KNeighborsClassifier

# instance model knn
my_knn = KNeighborsClassifier(n_neighbors=5)

# train the model
my_knn.fit(x_train, y_train)

# predict the class y_pred for x_test with our knn
y_pred = my_knn.predict(x_test)

# evaluate the score ofr presicion of y_pred with y_test

score_of_presicion = acuracy_score(y_test, y_pred)
print(f'The score of presicion is {score_of_presicion:.4f}')
```

### Prove the behavior of the model to predict new pacients

```py
df.info() # return the information of the dataframe
df.mean() # return the mean of the dataframe
df.median() # return the median of the dataframe (the most popular values)
df.std() # return the standard deviation of the dataframe
my_pacients = {
    'Grosor del grumo':[4,4 + 3, 4 - 3],
    'Uniformidad del tamaño de las células':[1, 1 + 3, 0],
    'Uniformidad de la forma de las célular': [1, 1 + 3, 0],
    'Adhesión marginal': [1,4,0],
    'Tamaño de una sola célula epitelial': [2, 2 + 2, 2 - 2],
    'Núcleos desnudos': [1, 1 + 4, 1 - 4],
    'Cromatina suave': [3, 3 + 2, 3 - 2],
    'Núcleos normales': [1, 1 + 3, 0],
    'Mitosis': [1, 1 + 2, 0],
    } # we are taking first a pacient with the median o average, then a pacient with most desviation and a pacient with less desviation this is to predict what kind of cancer they have

    X_test.head()
my_pacients = pd.DataFrame(my_pacients) # convert the dictionary to a dataframe
my_pacients # return the dataframe
df.info() # return the information of the dataframe

```

1. Lets prove with new pacients our model of machine learning KNN, making a new dataframe or setting a new csv

2. lets reescale the new dataframe

3. make the casification of the new pacients with our model KNN

```py
# lets reescale the pacients

my_pacients = my_scaler.transform(my_pacients) # apply the scaler

my_pacients # return the dataframe

my_pacients = pd.DataFrame(my_pacients, columns=my_cols) # recover the dataframe format of pandas in my_pacients

my_pacients # return the dataframe

clasification = my_knn.predict(my_pacients) # predict the class y_pred for my_pacients with our knn

clasification # return the median, the median aumented and the median reduced. First pacient is a bening cancer, second is maligne and the last is bening
```

## Resume

1. We load the base of data of pacients
2. We filter by columns
3. We create train data $X_{train}$ and test data $X_{test}, $y_{train}$ and $y_{test}$
4. We reescale the train data $X_{train}$ and test data $X_{test}$ to work well in the KNN model
5. We create our model KNN and train it with $X_{train}$ and $y_{train}$
6. We prove the eficiency of our KNN with $X_{test}$ and $y_{test}$
7. Our KNN model is ready and able to be apllied to new pacients

# ALL ABOUT KNN ORGANICED

---

### **K-Nearest Neighbors (KNN) - Breast Cancer Classification**

#### **Introduction to KNN**

KNN (K-Nearest Neighbors) is a supervised learning algorithm that classifies data based on the majority class of its nearest neighbors.

- **K** = Number of neighbors.
- KNN is a **lazy learning** algorithm (it does not build a model beforehand).
- KNN is **non-parametric** (does not assume any distribution).
- KNN is **non-linear**, **non-deterministic**, and **non-probabilistic**.

---

### **Example of KNN Classification**

- Given **K = 5**, the algorithm selects the **5 nearest neighbors**.
- If more neighbors belong to **class blue** than **class red**, the prediction will be **blue**.

---

## **1. Data Preprocessing**

We will use the **Breast Cancer Wisconsin** dataset for this classification task.

### **Load the Dataset**

```python
import pandas as pd

# Load the dataset
df = pd.read_csv("breast-cancer-wisconsin.csv", sep=",")

# Display 5 random samples
print(df.sample(5))

# Check unique values in 'Núcleos desnudos'
print(df['Núcleos desnudos'].value_counts())
```

### **Handling Missing Values**

The dataset contains missing values marked as `'?'`. We will remove these values.

```python
# Remove rows where 'Núcleos desnudos' is '?'
df = df[df['Núcleos desnudos'] != '?'].reset_index(drop=True)

# Convert 'Núcleos desnudos' to integer type
df['Núcleos desnudos'] = df['Núcleos desnudos'].astype(int)

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Separate input features (X) and target labels (y)
X = df.drop(columns=['Clase'])
y = df['Clase']
```

---

## **2. Splitting the Data**

We divide the dataset into **training (80%)** and **testing (20%)** sets.

```python
from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Check the shape of training and testing sets
print(X_train.shape, X_test.shape)
```

---

## **3. Feature Scaling**

Feature scaling improves model performance by normalizing the data.

```python
from sklearn.preprocessing import StandardScaler

# Save column names
feature_columns = X_train.columns

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train, columns=feature_columns)
X_test = pd.DataFrame(X_test, columns=feature_columns)
```

---

## **4. Training the KNN Model**

We train a **KNN classifier** with `K=5`.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Instantiate and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the KNN model is: {accuracy:.4f}")
```

---

## **5. Predicting for New Patients**

To test the model, we will classify three new patients:

- **One with average values**.
- **One with increased values (more deviation).**
- **One with decreased values (less deviation).**

### **Create a New Patients Dataset**

```python
# New patient data
new_patients = {
    'Grosor del grumo': [4, 4 + 3, 4 - 3],
    'Uniformidad del tamaño de las células': [1, 1 + 3, 0],
    'Uniformidad de la forma de las células': [1, 1 + 3, 0],
    'Adhesión marginal': [1, 4, 0],
    'Tamaño de una sola célula epitelial': [2, 2 + 2, 2 - 2],
    'Núcleos desnudos': [1, 1 + 4, 1 - 4],
    'Cromatina suave': [3, 3 + 2, 3 - 2],
    'Núcleos normales': [1, 1 + 3, 0],
    'Mitosis': [1, 1 + 2, 0],
}

# Convert to DataFrame
new_patients = pd.DataFrame(new_patients)

# Scale the new patient data
new_patients_scaled = scaler.transform(new_patients)
new_patients_scaled = pd.DataFrame(new_patients_scaled, columns=feature_columns)

# Predict classifications
predictions = knn.predict(new_patients_scaled)

# Print results
print("Predictions for new patients:", predictions)
```

- **First patient** → **Benign**
- **Second patient** → **Malignant**
- **Third patient** → **Benign**

---

## **6. Summary of the Process**

1. **Loaded the patient dataset**.
2. **Filtered and cleaned** unnecessary columns and missing values.
3. **Split the data** into training and testing sets.
4. **Scaled the features** to improve model performance.
5. **Trained a KNN model** with `K=5`.
6. **Evaluated model accuracy** on test data.
7. **Tested model predictions** with new patients.

# Sentiment Analysis

### This notebook aims to demonstrate the full process of text classification (sentiment analysis) on movie reviews.

#### The process followed is as follows:

1. Data loading
2. Rating distribution (Positive {>6} - Neutral {4–6} - Negative {<4})
3. Text normalization
4. Data splitting into training and test sets, creation of the bag-of-words model and its application to the texts
5. Classification model creation
6. Model evaluation
7. Use of the classification model on new records

```py
# Check if spaCy is installed and download the Spanish model if necessary
import spacy

try:
  nlp = spacy.load('es_core_news_sm')
except OSError:
  !python -m spacy download es_core_news_sm

# URL of the dataset containing movie reviews
url = 'https://drive.google.com/file/d/1m9nawSJI3SHNwlJaAP8ZFCFe0zE29LOe/view?usp=sharing'

# Extract file ID and generate direct download link
file_id = url.split('/')[-2]
url_download = 'https://drive.google.com/uc?id=' + file_id

# Load the compressed CSV file
df = pd.read_csv(url_download, compression='zip', sep='\\|\\|', engine='python')

# Display the first few records
print(df.head())

# Classification of sentiment polarity (Positive {>6}, Neutral {4–6}, Negative {<4})
def classify_polarity(review_rate):
    if 6 < review_rate <= 10:
        return 'Positive'
    elif 4 <= review_rate <= 6:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the polarity classification function to the dataset
df['polarity'] = df['review_rate'].apply(classify_polarity)

# Show results with polarity column
df

```

### Normalization of texts

- lets define as text a review_tittle and the review_text

```py
df['texto'] = df['review_title'] + ' ' + df['review_text']

# check te row 0
df.loc[0, ['review_title', 'review_text', 'texto']]
```

### change the data to a numpy array

```py
df_sentiments = df[['texto', 'polarity']] # select the columns

x = df_sentiments['texto'].to_numpy() # change the data to a numpy array
y = df_sentiments['polarity'].to_numpy() # change the data to a numpy array
```

### Save only relevang information

```py
import re

from tqdm import tqdm

# Importamos el modelo en español de spacy
nlp = spacy.load('es_core_news_sm')


def normalize(corpus):
    """Función que dada una lista de textos, devuelve esa misma lista de textos
       con los textos normalizados, realizando las siguientes tareas:
       1.- Pasamos la palabra a minúsculas
       2.- Elimina signos de puntuación
       3.- Elimina las palabras con menos de 3 caracteres (palabras que seguramente no aporten significado)
       4.- Elimina las palabras con más de 11 caracteres (palabras "raras" que seguramente no aporten significado)
       5.- Elimina las stop words (palabras que no aportan significado como preposiciones, determinantes, etc.)
       6.- Elimina los saltos de línea (en caso de haberlos)
       7.- Eliminamos todas las palabras que no sean Nombres, adjetivos, verbos o advervios
    """
    for index, doc in enumerate(tqdm(corpus)):
        doc = nlp(doc.lower())
        corpus[index] = " ".join([
            word.lemma_ for word in doc if (not word.is_punct)
            and (len(word.text) > 3)
            and (len(word.text) < 11)
             and (not word.is_stop)
            and re.sub('\s+', ' ', word.text)
            and (word.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV'])
        ])

    return corpus

    # apply the normalization
    X_norm = normalize(x)
    # X_norm = pd.DataFrame({'texto': X_norm})
    # X_norm

    # change to panda
    X_norm = pd.DataFrame({'texto': X_norm})
    X_norm
```

#### save X_norm

```py
import pickle

X_norm.to_pickle('X_norm.pkl') # save the normalized data to a pickle file

X_norm_2 = pd.read_pickle('X_norm.pkl') # load from peackle to pandas dataframe
X_norm_2
```

### divide partitions to test and train

```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
```

### We will perform a Bag of Words (BoW)

- Data partitioning for information related to the text.

```py
from sklearn.feature_extraction.text import CountVectorizer
my_bow = CountVectorizer(max_features=2000, min_df=3) # create the bag of words

X_train_bow = my_bow.fit_transform(X_train['texto']) # create the bag of words for train
X_test_bow = my_bow.transform(X_test['texto']) # create the bag of words for test
```

### We will select a machine of soport vector machine (SVM) as a classifier

```py
from sklearn.svm import LinearSVC

my_soport_vector_machine = LinearSVC() # create the soport vector machine

my_soport_vector_machine.fit(X_train_bow, y_train) # train the soport vector machine
```

1. Let's predict the **y_pred** the classification of **X_test_bow**
2. Let's calculatge the **accuracy** between **y_pred** and **y_test**

```py

y_pred = my_soport_vector_machine.predict(X_test_bow) # predict the classification of X_test_bow

y_pred # return the classification of X_test_bow
```

### calculate the percdentaje of accuracy betweeen y_pred and y_test of my soport vector machine

```py
from sklearn.metrics import accuracy_score

presition = accuracy_score(y_test, y_pred) # calculate the percdentaje of accuracy
print(f'Accuracy: {presition:.2f}')
```

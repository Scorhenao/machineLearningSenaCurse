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
**Important:** if you change the data from excel to a csv file the separation of each coma will be with , besides a ; it is called data *sesgada*

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
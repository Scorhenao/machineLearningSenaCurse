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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # calculate the percentage of precision
recall = recall_score(y_test, y_pred, average='weighted')  # calculate the percentage of recall
f1 = f1_score(y_test, y_pred, average='weighted')  # calculate the percentage of f1

# print results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1: {f1:.2f}')
```

#### conclution

- it seems that the model is not that condifent to clasify 50 - 50


### Use the model
```py
my_reviews = {
    "texto": [
        'La película El casillo Vagabundo me pareció muy interesante, me gustó el desarrollo de la protagonista y que pasara por cambios de edad de forma mágica. También disfruté los detalles que Studio Ghibli hace por todos los escenarios.',
        'Mi vecino Totoro Un clásico entrañable que captura la inocencia de la infancia y la magia de la naturaleza. Totoro se ha convertido en un ícono por una buena razón: es imposible no sonreír al verlo. La animación es cálida y encantadora, perfecta para toda la familia. es de estudio ghibli',
        'ferdinan  es un toro que lo crearon en un entornov muy macho pero el solo era libre con una planta entonces se bolo y vivio muy contento hasta que se incontro con el pasado y pudo ser libre listo',
        'Minecraft es una basura absoluta, una vergüenza sin alma que parece hecha por ejecutivos que jamás han agarrado un mouse en su vida, con una animación más falsa que el amor de ella, un guión tan vacío como el cerebro de petro, cero emoción, cero batallas épicas, ni siquiera un maldito Ender Dragon decente, solo diálogos basura y personajes que dan más pena que un creeper en modo pacífico, una película tan mala en toda la palabra',
        'orgullo y prejuicio  es mucho más que una historia de amor. Es una obra maestra de observación social y una exploración de los errores humanos que nos impiden ver la verdad de los demás y de nosotros mismos. El Lenguaje antiguo del siglo XIX es hermoso y el estilo de escritura es fresco, ingenioso y lleno de diálogos brillantes y entretenidos'
    ]
}

my_reviews = pd.DataFrame(my_reviews)

my_reviews

# normalizer
my_reviews_norm = normalize(my_reviews['texto'])
my_reviews    

my_reviews_bow = my_bow.transform(my_reviews_norm)
my_reviews_bow

my_clasification = my_soport_vector_machine.predict(my_reviews_bow)
my_clasification

my_reviews_norm.to_numpy()
```

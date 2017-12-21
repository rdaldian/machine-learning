# Importing Libraries
import pandas as pd

dataset = pd.read_csv("static/train_id_10.csv", encoding = "ISO-8859-1")

# Storing the sample text
n = 1
text_data = dataset.text.values[n]
print('\n','Original text:','\n', text_data)

# Preprocessing text
import re
text_preprocessed = re.sub('[^a-zA-Z0-9]', ' ', dataset['text'][n])
text_preprocessed = text_preprocessed.lower()
print("="*90)
print('\n','Preprocessed text:','\n', text_preprocessed)

# Tokenization
import nltk
# nltk.download('punkt')
text_list = nltk.word_tokenize(text_preprocessed)
print("="*90)
print('\n', 'Text tokenization:','\n', text_list)

# Stopwords
# stopwords in english
stopwords = nltk.corpus.stopwords.words('english')
# stopwords in bahasa
"""stopwords = []
stopwords_file = open('static/stopwords-id.txt').readlines() # Stopwords removal
for row in stopwords_file:
    stopwords.append(row.replace('\n', ''))"""
print("="*90)
print('\n', 'Stopwords:','\n', stopwords)

# Stemming
# stemming in english
"""from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()"""
# stemming in bahasa
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
sf = StemmerFactory()
stemmer = sf.create_stemmer()

# Cleaned text list
text_list_cleaned = [stemmer.stem(word) for word in text_list if word not in stopwords]
print("="*90)
print('\n', 'Final cleaned text:','\n', text_list_cleaned)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(text_list), len(text_list_cleaned)))
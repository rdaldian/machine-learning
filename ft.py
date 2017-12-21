# Importing Libraries
import numpy as np
import pandas as pd

# Importing the dataset
"""dataset = pd.read_csv("static/train_id.csv", encoding = "ISO-8859-1")"""
dataset = str(input("Input freetext: "))

# Cleaning the texts
import re
import nltk
# stopwords in bahasa
stopwords = []
stopwords_file = open('static/stopwords-ft.txt').readlines() 
for row in stopwords_file:
    stopwords.append(row.replace('\n', ''))
"""stopwords = nltk.corpus.stopwords.words('english') # stopwords in english"""
# Stemming
"""from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #stemming in bahasa"""
"""from nltk.stem.porter import PorterStemmer #stemming in english"""
corpus = []
for i in range(0, len(dataset)):
    """text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])"""
    text = re.sub('[^a-zA-Z]', ' ', dataset)
    text = text.lower()
    text = nltk.word_tokenize(text) # tokenization
    """sf = StemmerFactory()
    stemmer = sf.create_stemmer()"""
    """stemmer = PorterStemmer()"""
    """text = [stemmer.stem(word) for word in text if word not in stopwords]"""
    text = [word for word in text if word not in stopwords]
    """text = [stemmer.stem(word) for word in text]"""
    text = ' '.join(text)
    corpus.append(text)
    #print(corpus)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 200)
X = cv.fit_transform(corpus).toarray()

# Latent Dirichlet Allocation
from sklearn.decomposition import LatentDirichletAllocation
# define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*90)     
lda = LatentDirichletAllocation(n_components=1, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
lda.fit(X)
n_top_words = 3
print("\nSummaries in LDA model: ")
tf_feature_names = cv.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
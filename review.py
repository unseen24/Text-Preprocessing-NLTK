import pandas as pd
import nltk 
#nltk.download('punkt_tab')

from nltk import word_tokenize
from nltk import sent_tokenize

#load csv as a dataframe
df = pd.read_csv('C:\\Users\\Renz\\Desktop\\Python\\NLP\\tripadvisor_reviews.csv', encoding='cp1252')

#cleaning the review column
df_clean = df.dropna(subset=['review'])
review = df_clean['review'].tolist()

print(f'{review}')

#tokenize the review column
#apply tokenize and lowercase to the dataframe
sentences = df_clean['review'].apply(sent_tokenize).apply(lambda x: [s.lower() for s in x])
words = df_clean['review'].apply(word_tokenize).apply(lambda x: [w.lower() for w in x])
num_words = words.apply(len)
num_sentences = sentences.apply(len)

print(f'{sentences.head()}')
print(f'{words.head()}')
print(f'Number of words: {num_words.sum()}')
print(f'Number of sentences: {num_sentences.sum()}')


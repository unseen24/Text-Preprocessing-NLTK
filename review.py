import pandas as pd
import nltk 
import string
import re
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from bs4 import BeautifulSoup


#load csv as a dataframe
df = pd.read_csv('C:\\Users\\Renz\\Desktop\\Python\\NLP\\tripadvisor_reviews.csv', encoding='cp1252')

#cleaning the review column
df_clean = df.dropna(subset=['review'])

#tokenize the review column
#apply tokenize and lowercase to the dataframe
sentences = df_clean['review'].apply(sent_tokenize).apply(lambda x: [s.lower() for s in x])
words = df_clean['review'].apply(word_tokenize).apply(lambda x: [w.lower() for w in x])
num_words = words.apply(len)
num_sentences = sentences.apply(len)

'''
print(f'{sentences.head()}')
print(f'{words.head()}')

print(f'Number of words: {num_words.sum()}')
print(f'Number of sentences: {num_sentences.sum()}')
'''

#removing punctuations
no_punct = words.apply(lambda x: [w for w in x if w not in string.punctuation])
'''print(f'{no_punct.head()}')'''

#removing stop words
no_stop_words = no_punct.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
'''print(f'{no_stop_words.head()}')'''

#removing noise
no_noise = no_stop_words.apply(lambda x: BeautifulSoup(" ".join(x), 'html.parser').get_text())
no_noise = no_noise.apply(lambda x: re.sub(r'\d+', '', x))
no_noise = no_noise.apply(lambda x: re.sub(r'[^\w\s]', '', x) )
'''print(f'{no_noise.head()}')'''

#stemming
stemmer = PorterStemmer()
stemmed = no_noise.apply(lambda x: [stemmer.stem(w) for w in x])
'''print(f'{stemmed.head()}')'''

#lemmatization
#POS mapping helper function
def get_wordnet_pos(word):
    """Maps NLTK POS tag to WordNet POS tag format"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,  # 'a'
        'N': wordnet.NOUN, # 'n'
        'V': wordnet.VERB, # 'v'
        'R': wordnet.ADV   # 'r'
    }
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
lemmatized = no_noise.apply(lambda x: [lemmatizer.lemmatize(w, pos = get_wordnet_pos(w)) for w in x])

'''print(f'{lemmatized.head()}')'''

table = pd.DataFrame({
    'original': df_clean['review'],
    'tokenized': words,
    'cleaned': no_noise
    
})
print(table.head())
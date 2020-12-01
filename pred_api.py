# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# defing stopwords manually

model=load_model('D:/mod/cnn_toxic_new(1).h5')
with open('Downloads/tokenizer_toxic.pickle', 'rb') as handle:
    tokenizer_toxic = pickle.load(handle)

STOP_WORDS= set([ 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't",'utc'])


# initializzing for lemmatizing
wnl = WordNetLemmatizer()

# function for text cleaning
def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub("\S*\d\S*", "", str(x)).strip()
    x = re.sub('[^A-Za-z0-9]+', ' ', str(x))
    #Stop word removal and Applying WordNetLemmatizer
    x = ' '.join(wnl.lemmatize(w) for w in x.split() if w not in STOP_WORDS)
  
    return x

def tok_pad(text):
    sequences_pred = tokenizer_toxic.texts_to_sequences(text)
    sequ=[]
    seq_pred=[]
    for i in sequences_pred:
        for j in i:
            sequ.append(j)
    seq_pred.append(sequ)
    #padding 
    comment_pad=pad_sequences(seq_pred,maxlen=1250)
    return comment_pad    

def make_prediction(text):
    clean=preprocess(text)
    token=nltk.word_tokenize(clean)
    padded=tok_pad(token)
    predicted=model.predict(padded)
    predict_prob=[]
    for i in predicted:
        for j in i:
            j=j*100
            j=round(j,2)
            
            predict_prob.append(j)
    
    return(text,predict_prob)   


    


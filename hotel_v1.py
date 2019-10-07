# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:42:18 2019

@author: rogupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
stop=set(stopwords.words('english')) - set(['no','not','too','wont','nor'])
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
from nltk.tokenize import word_tokenize as token
lemmatizer = nltk.stem.WordNetLemmatizer()
eng_words = set(nltk.corpus.words.words())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
lr=LogisticRegression()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader=SentimentIntensityAnalyzer()
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
from string import digits
warnings.filterwarnings('ignore')
from wordcloud import WordCloud
#import pickle
#import json
#import os
#from flask import Flask,jsonify,request
#from flask_cors import CORS



#%% 

#contraction mapping

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def known_contractions(data):
    processed_text=[]
    #words = token(data)
    for w in data.split():
        x = contraction_mapping.get(w.lower(),w) 
        #get function will input as a word and return the value from the dictionary.
        #if we don't have the word in dictionary then it will return same value.
        processed_text.append(x)
    return ' '.join(processed_text)

#seperate postive and negative reviews in differen dataframe

def seperate(data):
    neg_data=data[['Negative_Review']]
    pos_data=data[['Positive_Review']]
    return pos_data,neg_data
    

#creating column for word count
def word_count(data):
    data['word_count']=data['Reviews'].str.split().str.len()
    return data

#data cleaning steps:
    
def lemme(data):
    text=[lemmatizer.lemmatize(w) for w in token(data)]
    return ' '.join(text)

def stop_words(data):
    text=[ w for w in token(data) if w not in stop]
    return ' '.join(text)
    
def duplicate_words(data):
    text=[w for w in token(data)]
    return ' '.join(sorted(set(text),key=text.index))

def spell_corrector(data):
    text=[spell(w) for w in token(data)]
    return ' '.join(text)

def get_nouns(data):
    text=token(data)
    pos_word=[word for word,pos in nltk.pos_tag(text) if ((pos=='NN') or (pos== 'JJ') or (pos== 'VB') or (pos== 'RB'))]
    return ' '.join(pos_word)

def remove_special(data):     
    good_symbols = re.compile('[^a-z]')     
    data = good_symbols.sub(' ',data)     
    return data

#https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b - to find codes for POS tags

def removing_single_letter(data):
    text=token(data)
    word=[w for w in text if len(w) >1]
    return ' '.join(word)    

def removing_double_letter(data):
    text=token(data)
    word=[w for w in text if len(w) >2]
    return ' '.join(word)    


def remove_non_english(data):
    text=token(data)
    non_eng=[w for w in text if w in eng_words]
    return ' '.join(non_eng)

def data_cleaning(data):
    data.loc[:,'reviews']=data['reviews'].apply(lambda x : x.lower()) #lower the character 
    data.loc[:,'reviews']=data.reviews.apply(remove_special)# remove special character
    data.loc[:,'reviews']=data['reviews'].replace({' +':' '},regex=True) # + means any number of space
    data.loc[:,'reviews']=data.reviews.apply(lemme)
    data.loc[:,'reviews']=data.reviews.apply(stop_words)
    data.loc[:,'reviews']=data.reviews.apply(duplicate_words)
    #data.loc[:,'reviews']=data.reviews.apply(get_nouns)
    data.loc[:,'reviews']=data.reviews.apply(removing_single_letter)
    data.loc[:,'reviews']=data.reviews.apply(removing_double_letter)
    #data.loc[:,'reviews']=data.reviews.apply(remove_non_english)
    data.loc[:,'reviews']=data['reviews'].apply(lambda x: x.strip()) # trim the space
    data1=data
    return data1

'''
#%% word cloud

pos_text = clean_data_final.reviews[clean_data_final['review_score']==1].str.cat(sep =' ')
neg_text = clean_data_final.reviews[clean_data_final['review_score']==0].str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud1 = WordCloud(width=1800,height=800,max_font_size=200).generate(neg_text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud1,interpolation='bilinear')
plt.axis("off")
plt.show()
'''


#%% applying tfidf

def tfidf_matrix(data):
    tfidf=TfidfVectorizer(min_df=5,ngram_range=(1,2))
    tfidf_matrix=tfidf.fit_transform(data)
    X_train_tfidf=tfidf_matrix.toarray()
    feature_names=np.array(tfidf.get_feature_names())
    return tfidf,X_train_tfidf,feature_names

def tfidf_test(data,tfidf):
    X_test_tfidf=tfidf.transform(data).toarray()
    return X_test_tfidf
    
#%%model

def model(model_name,X_train_tfidf,X_test_tfidf,y_train,y_test):
    model=model_name.fit(X_train_tfidf,y_train)
    pred=model.predict(X_test_tfidf)
    accuracy=accuracy_score(pred,y_test)
    return accuracy
    
#model_name,accuracy=model(lr,tfidf_array,X_test_tfidf,y_train,y_test)
#model_name,accuracy=model(lr,tfidf_array,X_test_tfidf,y_train,y_test)
#model_name,accuracy=model(nb,tfidf_array,X_test_tfidf,y_train,y_test)
def accuracy_model(final_score,model_name,model_list):
    for i in range(len(model_name)):
        accuracy_1=model(model_name[i],X_train_tfidf,X_test_tfidf,y_train,y_test)
        final_score=final_score.append({'model_name':[str(model_list[i])],'accuracy':[accuracy_1]},ignore_index=True)
    return final_score
    

#final_score=accuracy_score()

#final_score=pd.DataFrame({'model_name':[model_name],'accuracy':[accuracy]})
#print(final_score)



#%% calling of functions - main code
# reading the data
data=pd.read_csv('Hotel_Reviews.csv')

# function to seperate the data into positive n negative
pos_data,neg_data=seperate(data)

# renaming the columns
pos_data=pos_data.rename(columns={'Positive_Review' : 'Reviews'})
neg_data=neg_data.rename(columns={'Negative_Review' : 'Reviews'})


# adding word count columns 
pos_data=word_count(pos_data)
neg_data=word_count(neg_data)

#extract those rows which has count greater than equal to 3 in postive reviews
pos_data['review_score']=np.where(pos_data['word_count']>=3,1,0)
neg_data['review_score']=np.where(neg_data['word_count']>=3,0,1)

# ignoring the 0 in pos_data and 1 in neg_data
pos_data=pos_data[pos_data['review_score']==1]
neg_data=neg_data[neg_data['review_score']==0]

#concating pos and neg data which will be our final data
df=pd.concat([pos_data,neg_data],axis=0)
df=df.drop(['word_count'], axis=1)
df=df.drop_duplicates()
df=df.dropna()

#shuffle dataset
df=shuffle(df)

#renaming columns
df=df.rename(columns={'Reviews':'reviews'})

#creating sample for execution taking initial 10% data 
df1 = df.sample(frac=0.1, random_state=10)

#calling data_cleaning function    
clean_data=data_cleaning(df1)
clean_data=clean_data.drop_duplicates()
clean_data=clean_data.dropna()

#train test split
X=clean_data.loc[:,'reviews']
y=clean_data.loc[:,'review_score']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)


#calling tfidf 
tfidf,X_train_tfidf,feature_names=tfidf_matrix(X_train)
X_test_tfidf=tfidf_test(X_test,tfidf)

# calling accuracy model function
final_score=pd.DataFrame(columns=['model_name','accuracy'])
model_name=[lr,nb]
model_list =['Logistic Regression','MultinomialNB']

final_score = accuracy_model(final_score,model_name,model_list)


#%%


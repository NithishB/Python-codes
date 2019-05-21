import os
import time
import nltk
import string
import numpy as np
import pandas as pd
from textblob import TextBlob

nltk.download('all')

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def spellcheck(df, column):
    
    """
    The function corrects spellings of each word in that column
    
    Arguments:
    df - dataframe with textual data
    column - column name to use
    
    Returns:
    df - dataframe with corrected spellings (shape same as original)
    
    Example:
    >>> df = spellcheck(df, col_name)
    """
    for i in df.index:
        df[column][i] = TextBlob(df[column][i]).correct().raw
    return df

def clean(df, tasks, columns):
    
    """
    The function performs various cleaning tasks from lower case conversion to tokenization and lemmatization. 
    Need to pass a dataframe of textual data for cleaning.
    
    Arguments:
    df - pandas dataframe with textual data to clean
    tasks - list of tasks to perform. 
            '- ' or '-_' replaces '-' with spaces or underscores resp. 
            'punct' removes all punctuations from the data.
            'num' replaces all digits with '#'.
            'lemma' performs lemmatization on words.
            'stop' removes stopwords from data.
            'spell' runs textblob spellcheck.
    columns - list of columns in dataframe to clean.
    
    Returns:
    df - cleaned dataframe (same shape as original)
    
    Example:
    >>> df1 = clean(df,task_list,col_list)
    """
    
    for column in columns:
        #Lowercase conversion
        df[column] = df[column].apply(lambda x: x.lower())
        print(column+": Converted to lowercase")

        #Tokenization
        df[column] = df[column].apply(word_tokenize)
        df[column] = df[column].apply(lambda x: " ".join(x))
        print(column+": Tokenized")

        if('- ' in tasks):
        #Split a-b into a and b
            df[column] = df[column].str.replace('-',' ')
            print(column+": - Replaced")

        elif('-_' in tasks):
        #Split a-b into a and b
            df[column] = df[column].str.replace('-','_')
            print(column+": - Replaced")

        if('punct' in tasks):
        #Removing punctuations
            df[column] = df[column].str.replace('[^\w\s]','')
            print(column+": Removed punctions ")

        if('num' in tasks):
        #Replacing numbers
            df[column] = df[column].str.replace('[0-9]','#')
            print(column+": Replaced Numbers ")

        if('stop' in tasks):
        #Removing Stop Words
            df[column] = df[column].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
            print(column+": StopWords Removed")

        if('lemma' in tasks):
        #Lemmatization - root words
            df[column] = df[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word,pos='v') for word in x.split()]))
            print(column+": Root words Lemmatized")

        if('spell' in tasks):
        #Spellcheck words
            df = spellcheck(df, column)
            print(column+": Word spellings corrected")
    
    print("Null values:",df.isnull().values.any())
    return df
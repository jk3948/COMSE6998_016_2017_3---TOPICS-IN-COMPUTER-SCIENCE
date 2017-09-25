#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:33:09 2017

@author: jingkong
"""

import sklearn
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

thePath = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/classify/'

import os 

theCols = os.walk(thePath).next()[1]

theLabels = theCols

finalWords = list()


def genCorpus(theText):
#set dictionaries
    stopWords = set(stopwords.words('english'))#english 里的stopwords,like filter
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    #ex:running,runner...==run
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces
    
    return tokens


for words in theCols:
    d = dict()
    #cnt = 0
    for file in os.listdir(thePath + words):#find files and load files
        if file.endswith('.txt'):
            try:
                f = open(thePath + words + "/" +file, "r") 
                lines = f.readlines()#scan all the text
                lines = [text.strip() for text in lines]#strip clean the text
                lines = " ".join(lines)#concate arrays 
                finalWords.append(genCorpus(lines))
            except:
                pass
    temp=list()
    for j in range(len(finalWords)):
        temp = temp + finalWords[j].split(" ")
        
    temp_uniq = list(set(temp))
    
    
    #temp = list(set(finalWords[j for j in range(len(finalWords))].split(" ")))
    for i in temp_uniq:
        count = temp.count(i)
        d[i] = count
    
    df = pd.DataFrame.from_dict(d,orient='index')
    df.index.name='word'
    df.columns = ['freq']
    df.sort_values('freq', axis = 0, ascending=False)
    df.to_csv(thePath + words + ".csv")






























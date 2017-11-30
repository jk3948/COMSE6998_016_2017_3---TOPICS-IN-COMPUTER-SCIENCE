#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:33:10 2017

@author: jingkong
"""
import pandas as pd
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import r2_score, recall_score, precision_score, accuracy_score
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import ast
import pandas as pd
from sklearn.externals import joblib
import sys
import nltk
from nltk.corpus import stopwords
import re
import os
import tweepy  
from pymongo import MongoClient
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
from bson.json_util import dumps
json = import_simplejson()


def genCorpus(theText):
    #set dictionaries
    stopWords = set(stopwords.words('english'))
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       

    return tokens

##Pre-processing and data cleaning
thePath = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/hw/hw4/'
df = pd.read_csv(thePath + "private us companies.csv" )
df=df.iloc[:,1:3]
y = df.iloc[:,1]
X = df.iloc[:,0]
theCols = y.unique()
theLabels = theCols 
finalWords = list()
theDocs = list()
myFun = classA()
for word in theCols:
    temp = df[y==word]
    cnt = 0
    if len(temp) > 100:        
        ind = np.random.choice(temp.index, 100,replace=False)##undersampling
        arr = df.iloc[ind,0]
    else:
        ind = np.random.choice(temp.index, 100,replace=True)##oversampling
        arr = df.iloc[ind,0]
    for lines in arr:
        try:
            
            finalWords.append(myFun.genCorpus(lines))
            theDocs.append(myFun.textToNum(theLabels,word) +"_" + str(cnt))
            cnt = cnt +  1
        except:
            pass    

tdm = myFun.vec(finalWords,1000,1,1,theDocs)
reducedTDM = myFun.pca(tdm,0.95,theDocs)

fullIndex = reducedTDM.index.values
fullIndex = [int(word.split("_")[0]) for word in fullIndex]
##train models
myFun.modelTrain(['RF'],reducedTDM,fullIndex,10,thePath)
vectorizer = joblib.load(thePath + 'vectorizer.pk') 
pca = joblib.load(thePath + 'pca.pk') 
for file in os.listdir(thePath):
    if file.endswith(".pkl"):
        theFile = file
model = joblib.load(thePath + theFile)

##Classify arbitrary text for specific industry
dict1 = dict()
for tempText in X:
    
    testText = list()
    
    testText.append(genCorpus(tempText))
    test = vectorizer.transform(testText) 

    X2_new = pca.transform(test.toarray())
    x = model.predict(X2_new)
    xProba = pd.DataFrame(model.predict_proba(X2_new))#predict_proba gives you all the posterior prob
    xProba = xProba.round(4) 
    xProba.columns=theCols[0:]
    xProba = xProba.to_json()
    sys.stdout.write(xProba)
    sys.stdout.write('\n')
    xProba = ast.literal_eval(xProba)
    keyword = max(xProba, key=xProba.get)
    
    try:     
        '''
        print( keyword )
        
        mongo_collection.insert({
        'message_id': testText.index,
        
        'body': testText,
        'topic': keyword,

        })
        '''
        dict1[tempText] = keyword
    except Exception, (e):  
        print("HERE")          
        pass 

df2 = pd.DataFrame(dict1.items())    
    
col_name = ['Company','Predicted Industry']
df2.columns = col_name    
df2.to_csv('classfication.csv')    
    
    
    
    
    
    

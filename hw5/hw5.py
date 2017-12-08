#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:07:16 2017

@author: jingkong
"""
import collections
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
import json
## Prerprocessing
thePath = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/hw/hw5/'
df = pd.read_csv(thePath + "user_ages_train.csv")
df.index = df.ID
df_test = pd.read_csv(thePath + "user_ages_test.csv")

#df_f = pd.read_csv(thePath + "friends.csv" )
#set age range
df['age_range'] =0
for i in range(len(df)):
    if 18 <= df.iloc[i,1] < 25:
        df['age_range'].iloc[i]='18-24'
    if 25 <= df.iloc[i,1] < 35:
        df['age_range'].iloc[i]='25-34'
    if 35 <= df.iloc[i,1] < 45:
        df['age_range'].iloc[i]='35-44'
    if 45 <= df.iloc[i,1] < 55:
        df['age_range'].iloc[i]='45-54'
    if 55 <= df.iloc[i,1] < 65:
        df['age_range'].iloc[i]='55-64'
    elif 65 <= df.iloc[i,1]:
        df['age_range'].iloc[i]='65+'


#read 2 json files and preprocess
with open(thePath +'user_age_profiles.json') as json_data:
    uap = json.load(json_data)
    

with open(thePath +'user_age_tweets.json') as json_data:
    uat = json.load(json_data)
    

'''
uap:
"profile_link_color"
"name": "Greggy"
"profile_text_color"
"id"
"friends_count"
"default_profile_image": false,
"profile_use_background_image": true
"followers_count": 274,

"status": {"text"
retweet_count": 452
"description"
"followers_count"
text
in_reply_to_user_id_str"
id"


uat:
id(tweet id)
"in_reply_to_status_id": null,
"text"
"user": {
followers_count
retweet_count'
favorite_count
id
geo_enabled
name
description'
         
'''
def createTable(df):       
    temp = []
    for l in range(len(uap)):
         
        temp.append(uap[l]['id'])
    
    temp2 = [] 
    for l in range(len(uat)):
         
        temp2.append(uat[l]['user']['id'])
        
    li = list(set(df['ID']).intersection(set(temp)).intersection(set(temp2)))
    
    df2 = pd.DataFrame(columns=['age','id','name','friends_count','profile_text_color','default_profile_image', 
                   'profile_use_background_image','text','description','followers_count', 
                   'retweet_count','favorite_count','geo_enabled'],index = li)
    
    #df2.dtypes           
    for id in li:
        for j in range(len(uap)):           
            if uap[j]['id'] == id:                
                try:
                    df2.age[id] = df.age_range[id]
                except:                
                    pass
                try:
                    
                    df2.id[id]=id
                    df2.name[id]=uap[j]['name']
                    df2.friends_count[id]=uap[j]['friends_count']
                    df2.profile_text_color[id]=uap[j]['profile_text_color']
                    df2.default_profile_image[id]=uap[j]['default_profile_image'] 
                    df2.profile_use_background_image[id]=uap[j]['profile_use_background_image']
                    df2.text[id]=uap[j]['status']['text']
                    df2.description[id]=uap[j]['description']
                    df2.followers_count[id]=uap[j]['followers_count']
                    df2.retweet_count[id]=uap[j]['status']['retweet_count']
                except: 
                    pass                      
        for k in range(len(uat)):
            if id == uat[k]['user']['id']:
                try:
                    df2.favorite_count[id]=uat[k]['favorite_count']
                    df2.geo_enabled[id]=uat[k]['user']['geo_enabled']
                except:
                    pass
    return df2
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
    
df2 = createTable(df)          
df3 = df2.dropna(0,how="any")  
df2_test = createTable(df_test)
df3_test = df2_test.drop(['age'],axis=1).dropna(0,how="any")  
##deal with train text data
df_text = pd.DataFrame(df3['name']+'_'+df3['text']+'_'+df3['description'])
df_text['text']=df_text[0]
df_text=df_text.drop([0],axis=1)
df_text['age']=df3.age
y = df3.age
theCols = y.unique()
theLabels = theCols 
finalWords = list()
theDocs = list()
myFun = classA()
for word in theCols:
    temp = df_text[df_text.age==word]['text']
    cnt = 0
    for lines in temp:
        try:            
            finalWords.append(myFun.genCorpus(lines))
            theDocs.append(myFun.textToNum(theLabels,word) +"_" + str(cnt))
            cnt = cnt +  1
        except:
            pass    
tdm = myFun.vec(finalWords,1000,1,1,theDocs)

##deal test text data
df_text_test = pd.DataFrame(df3_test['name']+'_'+df3_test['text']+'_'+df3_test['description'])
df_text_test['text']=df_text_test[0]
df_text_test=df_text_test.drop([0],axis=1)


finalWords_test = list()
for lines in df_text_test.text:
    try:            
        finalWords_test.append(myFun.genCorpus(lines))
        
    except:
        pass    
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,1))
tdm_test = pd.DataFrame(vectorizer.fit_transform(finalWords_test).toarray())

##deal with category variables and train model
df3_comb =pd.concat([df3,df3_test],axis=0)
df3_comb_cat=pd.get_dummies(df3_comb, columns=['geo_enabled',"profile_text_color",'default_profile_image','profile_use_background_image','profile_use_background_image'], prefix=['geo_enabled',"profile_text_color",'default_profile_image','profile_use_background_image','profile_use_background_image'])
df3_cat = df3_comb_cat.iloc[0:1541,:]
df3_cat_test = df3_comb_cat.iloc[1541:,:]
df4_cat = df3_cat.drop(['description','id','index','name','text'],axis=1).reset_index()
df4_cat_test = df3_cat_test.drop(['description','id','index','name','text'],axis=1).reset_index()
tdm_4 = tdm.reset_index()
df_trainf = pd.concat([df4_cat,tdm_4],axis=1)

y_train = df_trainf.age
tdm_4_test = tdm_test.reset_index()
df_testf = pd.concat([df4_cat_test,tdm_4_test],axis=1)
df_trainf=df_trainf.drop(['index','age'],axis=1).dropna(0,how='all')
X_train= df_trainf
df_testf=df_testf.drop(['index','age'],axis=1).dropna(0,how='all')
X_test= df_testf

rf = RandomForestClassifier(n_estimators=50, random_state=50)
model = rf.fit(X_train, y_train)

y_predicted = rf.predict(X_test)

df_pred = pd.DataFrame()    
df_pred['ID']=df3_test.index   
df_pred['age_prediction']=y_predicted     
df_pred.to_csv('age_predictions')    























































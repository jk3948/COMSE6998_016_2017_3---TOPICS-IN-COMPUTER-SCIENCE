#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:15:19 2017

@author: jingkong
"""
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

auth1 = tweepy.auth.OAuthHandler('7SsrQmhFuO96HSJ44PVzjHvax','vyDh4OVdd1dKdowsokaOnwfDTmTfdnkk64yjnEkn0ThhT68GHs')  #twitter consumer key
auth1.set_access_token('814416506-QoCWazbeKaJFgSGAnA7FKeHxpwNWbj1oFRQX9VIc','ufB00E8F7rozFOvk4O2cUQLE2atrdHdeJDSTn5Cs0dqKt')  
api = tweepy.API(auth1)#api call

mongo = MongoClient('localhost', 27017)
mongo_db = mongo['twitterDBs']#creat database
mongo_collection = mongo_db['theData2']#same as sql tables,mongo has collections

thePath = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/classify/'
path = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/classify/' 

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

theCols = os.walk(thePath).next()[1] 

vectorizer = joblib.load(path + 'vectorizer.pk') 
pca = joblib.load(path + 'pca.pk') 
for file in os.listdir(path):
    if file.endswith(".pkl"):
        theFile = file
model = joblib.load(path + theFile)

class StreamListener(tweepy.StreamListener):  
    status_wrapper = TextWrapper(width=140, initial_indent='', subsequent_indent='')    
    def on_status(self, status): 
        tempA = self.status_wrapper.fill(status.text)
        tempB = status.retweeted 
        tempC = status.user.lang 
        tempD = status.geo
        
        testText = list()
        testText.append(genCorpus(self.status_wrapper.fill(status.text)))
        test = vectorizer.transform(testText)
        X2_new = pca.transform(test.toarray())
        x = model.predict(X2_new)
        xProba = pd.DataFrame(model.predict_proba(X2_new))#predict_proba gives you all the posterior prob
        xProba = xProba.round(4) 
        xProba.columns=theCols[1:]
        xProba = xProba.to_json()
        sys.stdout.write(xProba)
        sys.stdout.write('\n')
        xProba = ast.literal_eval(xProba)
        keyword = max(xProba, key=xProba.get)
           
       
        try:     
            print(self.status_wrapper.fill(status.text), keyword )
            
            mongo_collection.insert({
            'message_id': status.id,
            'screen_name': status.author.screen_name,
            'body': self.status_wrapper.fill(status.text),
            'topic': keyword,
            'created_at': status.created_at,
            'followers': status.user.followers_count,
            'friends_count': status.user.friends_count,
            'location': status.user.location
            })
            
        except Exception, (e):  
            print("HERE")          
            pass 
print "here" 
l = StreamListener()  
streamer = tweepy.Stream(auth=auth1, listener=l, timeout=3000)   
setTerms = ["fishing","hiking","machine learning","mathematics"]
streamer.filter(None,setTerms)


##Export json 
##run the following code directly from the system command 
#mongoexport  --db twitterDBs --collection theData2 --out sample.json


























#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:25:24 2017

@author: jingkong
"""
import pandas as pd
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
import google
import ssl
import urllib2
import urllib
from bs4 import BeautifulSoup
import os
import errno


thePath = '/Users/jingkong/Dropbox/Columbia/Fall2017/project datasci/hw/hw3/'
fileIndex = list()
theQuery = ["politics", "astronomy", "medical", "music", "sports"]



def crawler(theQuery):    
    for url in google.search(theQuery, num=200, start=0, stop=20):
        fileIndex.append(url)
    
    cnt = 0
    for theUrl in fileIndex:
        try:
        #for i in range(0,1):
            #theUrl = fileIndex[i]
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib2.build_opener(urllib2.HTTPSHandler(context=ctx))
            opener.addheaders = [('Referer', theUrl)]
            html = opener.open(theUrl,timeout=10).read()
            soup = BeautifulSoup(html,"lxml")
            
            textTemp = list()
            try:
                textTemp.append(soup.find('title').text)
                textTemp.append('\n')
                for theText in soup.find_all(['p'],text=True): #,'li']):#,'li']):#,'ul']):#,'span']):#,'li']):
                    textTemp.append(theText.text)
            except:
                print theUrl
                pass    
        
            text = " " . join(textTemp)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            if len(text) >= 50:                
                tmpFile = str(cnt) + ".txt"
                if not os.path.exists(os.path.dirname(thePath + theQuery+ "/" + tmpFile)):
                    try:
                        os.makedirs(os.path.dirname(thePath + theQuery+ "/" + tmpFile))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                indexFile = open(thePath + theQuery+ "/" + tmpFile, "w")
                indexFile.write(text.encode('utf8'))
                indexFile.close()
                cnt = cnt + 1
            else:
                pass
        except:
            pass
    
    
#theQuery = ["politics"]
for i in theQuery:
    crawler(i)    



theCols = os.walk(thePath).next()[1]  

theLabels = theCols 

finalWords = list()
theDocs = list()
    
myFun = classA()

for word in theCols:
    cnt = 0
    for file in os.listdir(thePath+word):
        if file.endswith('.txt'):
            try:
                f = open(thePath + word + "/" + file, "r")
                lines = f.readlines()
                lines = [text.strip() for text in lines]
                lines = " ".join(lines)
                finalWords.append(myFun.genCorpus(lines))
                theDocs.append(myFun.textToNum(theLabels,word) +"_" + str(cnt))
                cnt = cnt +  1
            except:
                pass

tdm = myFun.vec(finalWords,1000,1,1,theDocs)

reducedTDM = myFun.pca(tdm,0.95,theDocs)

fullIndex = reducedTDM.index.values
fullIndex = [int(word.split("_")[0]) for word in fullIndex]

myFun.modelTrain(['RF'],reducedTDM,fullIndex,10,thePath)


    
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
mongo_collection = mongo_db['theData3']#same as sql tables,mongo has collections


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


vectorizer = joblib.load(thePath + 'vectorizer.pk') 
pca = joblib.load(thePath + 'pca.pk') 
for file in os.listdir(thePath):
    if file.endswith(".pkl"):
        theFile = file
model = joblib.load(thePath + theFile)

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
        xProba.columns=theCols[0:]
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
setTerms = ["potus", "moon and the sun", "pharmacy", "drake", "quarterback"]
streamer.filter(None,setTerms)

##Export json 
##run the following code directly from the system command 
#mongoexport  --db twitterDBs --collection theData3 --out hw3.json    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
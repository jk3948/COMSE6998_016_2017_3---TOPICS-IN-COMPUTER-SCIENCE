#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:44:11 2017

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

class classA(object):
    
    def genCorpus(self,theText):
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
        self.tokens = " ".join(tokens) #need to pass string seperated by spaces       
    
        return self.tokens
    
    def textToNum(self,theLabels,thePredLabel):
        theOutLabel = dict()
        cnt = 0
        for word in theLabels:
            theOutLabel[word] = cnt
            cnt = cnt + 1
        return str(theOutLabel[thePredLabel])    
    
    def algoArray(self,theAlgo,thePathLut):
        self.theLUT = pd.read_csv(thePathLut + 'classifierLUT.csv',index_col=0) #ALGO LUT
        theAlgoOut = self.theLUT.loc[theAlgo,'functionCall']
        return theAlgoOut

    def vec(self,varData,maxF,nGramMin,nGramMax,theDocs):
        vectorizer = TfidfVectorizer(max_features=maxF,ngram_range=(nGramMin,nGramMax))
        tdm = pd.DataFrame(vectorizer.fit_transform(varData).toarray())
        
        with open('vectorizer.pk', 'wb') as fin:
            pickle.dump(vectorizer, fin)
    
        tdm.columns=vectorizer.get_feature_names()
        tdm.index=theDocs
        
        return tdm

    def pca(self,varData,varExp,theDocs):
        pca = decomposition.PCA(n_components=varExp)
        pca.fit(varData)
        reducedTDM = pd.DataFrame(pca.transform(varData)) #reduced tdm distance matrix
        
        with open('pca.pk', 'wb') as fin:
            pickle.dump(pca, fin)
        
        reducedTDM.index=theDocs
    
        return reducedTDM

    def optFunc(self,theAlgo,theParams):
        theModel = self.theLUT.loc[theAlgo,'optimizedCall']
        tempParam = list()
        for key, value in theParams.iteritems():
            tempParam.append(str(key) + "=" + str(value)) 
        theParams = ",".join(tempParam)
        theModel = theModel + theParams + ")"
        return theModel 

    def modelTrain(self,modAr,redTdm,fullInd,fld,thePathLut):
        self.fullInd = fullInd
        theModels = modAr#,'ABDT']#,'LOGR']#,'NN']#,'DT','ABDT','LDA']#,'DT','LDA','BAG','KNN','NN'] #these MUST match up with names from LUT #ABDT, #GBC, #RSM take far too long
        theResults = pd.DataFrame(0,index=theModels,columns=['accuracy','confidence','runtime'])
        for theModel in theModels:
            startTime = time.time()
            model = eval(self.algoArray(theModel,thePathLut))
            print(theModel)
        
            #cross validation    
            cvPerf = cross_val_score(model,redTdm,self.fullInd,cv=fld)
            theResults.ix[theModel,'accuracy'] = round(cvPerf.mean(),2)
            theResults.ix[theModel,'confidence'] = round(cvPerf.std() * 2,2)
            endTime = time.time()
            theResults.ix[theModel,'runtime'] = round(endTime - startTime,0)
            
        print(theResults)
        
        #############################################
        #######Run with best performing model########
        #####Fine Tune Algorithm Grid Search CV######
        #############################################
        bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
        modelChoice = theResults['accuracy'].idxmax()
                      
        startTime = time.time()
        model = eval(self.algoArray(modelChoice,thePathLut))
        grid = GridSearchCV(estimator=model, param_grid={"n_estimators": [10,30,50,100]})#eval(gridSearch(modelChoice))
        grid.fit(redTdm,self.fullInd)
        #grid.fit(train,trainIndex)
        bestScore = round(grid.best_score_,4)
        parameters = grid.best_params_
        endTime = time.time()
        print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime,0)))
        
        ############################################
        ######Train Best Model on Full Data Set#####
        ########Save Model for future use###########
        ############################################
        startTime = time.time()
        model = eval(self.optFunc(modelChoice,parameters)) #train fully validated and optimized model
        model.fit(redTdm,self.fullInd)
        #model.fit(train,trainIndex)
        joblib.dump(model, modelChoice + '.pkl') #save model
        endTime = time.time()
        print("Model Save Time: " + str(round(endTime - startTime,0)))
        
        return None
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:13:00 2020

@author: richa
"""
#predict if review is positive or negative for any new review
#using tsb delimiter tab coz reviews already contain commas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#clean the dataset get rid of words that are not useful like the on at or numbers
#apply steming convert into same tense to reduce words with same meaning
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review =review.lower()
    review=review.split()
    ps=PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X=cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Fitting SVM to the Training set

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







    
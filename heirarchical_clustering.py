# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:31:59 2020

@author: richa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

#using dendogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
denogeam = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distances')
plt.show()

#fit model to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#visualizing clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='sensible')
plt.xlabel('Annual income')
plt.ylabel('Spending SCore')
plt.legend()
plt.show()
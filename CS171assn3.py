#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:43:48 2018

@author: Brettmccausland
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator
from operator import itemgetter, attrgetter
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

def minkowskiDist(R,Rn,p):
  sum=0
  count=len(R)
  for i in range(count):
    sum = sum + pow((abs(R[i]-Rn[i])),p)
  fp= 1/p
  return pow(sum,fp)
#performance measures:
def accuracy(ypred,ytest):
 c = 0
 for i in range(len(ytest)):
     if(ytest[i] == ypred[i]):
         c += 1
 return(c/float(len(ytest)) * 100.0)

def Rowneighbors(X_train, X_test,k,p):
   d = []
   neighbors = []
   trainsize = len(X_train)

   for i in range(trainsize):
     dist = minkowskiDist(X_train[i],X_test,p)
     d.append((dist,i))
   #sort by dist
   d.sort(key=operator.itemgetter(0))
   for i in range(k):
     neighbors.append(d[i])
   return neighbors
def k_means(X_input, K,centroids):
  
  clusterCentroids=centroids
  numRows=X_input.shape[0]
  print(numRows)
  
#this is a (data point x 1) vector that contains 
#the cluster number that each data point is assigned to.
  clusterAssignmets=[]
# Partition the data at random into k sets
  for i in range(numRows):
    clusterAssignmets.append(i%k)
  for k
    # Calculate the centroid of each sets
# assign each point to the set corresponding to the closest centroid
# Repeat the last two steps until nothing is moved around, or some max iteration
#   has been achieved
  print(clusterAssignmets)
  return

#----------- Question 1: Feature distribution ----------
# 1)
   #import the data set
data=pd.read_csv('IrisDataSet.csv')
   #shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
   #feature matrix input
X_input = data.iloc[:,:-1]    
Y_input = data.iloc[ :, -1:]   
k=3
p=2
centroids=X_input.iloc[0:k,:]
k_means(X_input, k,centroids)
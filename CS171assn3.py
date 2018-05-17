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
def assignClassification(centroids,X_input):
 # assign each point to the set corresponding to the closest centroid
 K=centroids.shape[0]
 cAssign = [0] * K
 numRows=X_input.shape[0]
 d=[]
 for row in range(numRows):
    for k in range(K):
        dist= minkowskiDist(centroids[k],X_input[row],2)
        d.append([dist,k])
    d.sort(key=operator.itemgetter(0))
    cAssign[row]=d[0].itemgetter(1)
    d.clear()
 return
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

#calculate the centroids with meancluster 
#add all fetures/columns together with same cluster values
#divide colums by the number of occurances k_count
def meancluster(centroids,X_input,cAssign ):
 print('input to meancluster')
 print('centroids',centroids)
 print('centroids',X_input)
 print('cAssign',cAssign)
 
 K=centroids.shape[0]
 k_counts = [0] * K
 numRows=X_input.shape[0]
 numCol=X_input.shape[0]
 meancluster=centroids
 meancluster[:] = 0
 
 for i in range(numRows):
   for j in range(numCol):
     meancluster[cAssign[i],j]+= X_input[i,j]
     k_counts[cAssign[i]]+=1
 for k in range(K):
   for j in range(numCol):
     meancluster[k,j]= meancluster[k,j]/k_counts[k]
     centroids=meancluster
   meancluster[:] = 0
   k_counts.fill(0)
 return
def k_means(X_input, K,centroids):
  
  print('inside k means')
  meancluster=centroids
  meancluster[:] = 0
  print('meancluster:',meancluster)
  k_counts = [0] * K
  print('k_counts:', k_counts)
  numRows=X_input.shape[0]
  print('numRows:', numRows)
  numCol=X_input.shape[1]
  print('numCol:', numCol)
  max_iteration=30
  
  d=[]
  #this is a (data point x 1) vector 
  cAssign=[]

# Partition the data at random into k sets
  for i in range(numRows):
    cAssign.append(i%K)
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
    assignClassification(centroids,X_input)
    meancluster(centroids,X_input,cAssign )
    print(cAssign)
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
  K=3
  centroids=X_input.iloc[0:K,:]
  k_means(X_input, K,centroids)
  print('hello')
  
 
  meancluster=centroids
  meancluster[:] = 0
  k_counts = [0] * K
  
  numRows=X_input.shape[0]
  numCol=X_input.shape[1]
  max_iteration=30
  print(numRows)
  d=[]
#this is a (data point x 1) vector that contains 
#the cluster number that each data point is assigned to.
  cAssign=[]

# Partition the data at random into k sets
  for i in range(numRows):
   cAssign.append(i%K)
  print('hello')
  dist=0
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
 # assign each point to the set corresponding to the closest centroid
   for row in range(numRows):
        for k in range(K):
            #print('centroids',centroids)
            #print('X_input[row]',X_input.iloc[row,:])
            #c=list(centroids.iloc[0])
            #print(c)
            #dist= minkowskiDist(list(centroids[k]),list(X_input[row]),2)
            dist= minkowskiDist(list(centroids.iloc[k]),list(X_input.iloc[row]),2)
            d.append([dist,k])
        #d.sort(key=operator.itemgetter(0))
        #cAssign[row]=d[0].itemgetter(1)
        d.clear()
#calculate the centroids with meancluster  
  #add all same type cluster values
  #add one to the k_count
   for i in range(numRows):
     for j in range(numCol):
       meancluster[cAssign[i],j]+= X_input[i,j]
   k_counts[cAssign[i]]+=1
   #divide colums by the number of occurances k_count
   for k in range(K):
     for j in range(numCol):
       meancluster[k,j]= meancluster[k,j]/k_counts[k]
   centroids=meancluster
   print('hello')
   meancluster[:] = 0
   print('hello agian')
   k_counts.fill(0)'''
   
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

# assign each point to the set corresponding to the closest centroid
def assignClassification(centroids,X_input):
 
 # assign each point to the set corresponding to the closest centroid
 K=centroids.shape[0]
 #print(centroids)
 cAssign=[]
 numRows=X_input.shape[0]
 d=[]
 
 #create assignment vector of correct size
 #initialized at random
 for i in range(numRows):
   cAssign.append(i%K)

 for row in range(numRows):
   for k in range(K):
     dist= minkowskiDist(list(centroids.iloc[k]),list(X_input.iloc[row]),2)
     d.append([dist,k])
   d.sort(key=operator.itemgetter(0))
   best=d[0]
   cAssign[row]=best[1]
   d.clear()
 print('cAssign',cAssign)
 return cAssign
#calculate the centroids with meancluster 
#add all fetures/columns together with same cluster values
#divide colums by the number of occurances k_count
def meancluster(centroids,X_input,cAssign ):

 K=centroids.shape[0]
 k_counts=[]
 for k in range(K):
   k_counts.append(0)
 numRows=X_input.shape[0]
 numCol=X_input.shape[1]
 print('cAssign',cAssign)
 mc=centroids
 mc[:] = 0

 #talle column values and take k count
 for row in range(numRows):
   Xrow= list(X_input.iloc[row])
   Addr= cAssign[row]
   MCrow=list(mc.iloc[Addr])
   k_counts[Addr]+=1
   for col in range(numCol):
     Sum= Xrow[col]+ MCrow[col]
     print('Sum',Sum)
     mc.iloc[Addr,col]=Sum
   #print(k_counts)
 for row in range(K):
   for col in range(numCol): 
     value=mc.iloc[row,col]
     print('value',value)
     D=k_counts[row]
     print('D',D)
     mc.iloc[row,col]= (value/D)
 centroids=mc
  
 return
def k_means(X_input, K,centroids):
  
  max_iteration=5
  #this is a (data point x 1) vector 
  cAssign=[]
  
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
    cAssign=assignClassification(centroids,X_input)
    meancluster(centroids,X_input,cAssign )
  return cAssign

#----------- Question 1: k-Means----------
# 1)
   #import the data set
  data=pd.read_csv('IrisDataSet.csv')
   #shuffle the data
  data = data.sample(frac=1).reset_index(drop=True)
   #feature matrix input
  X_input = data.iloc[:,:-1]    
  #Y_input = data.iloc[ :, -1:]   
  K=3
  centroid=X_input.iloc[0:K,:]
  a= X_input.iloc[0,1]
  
  cAssign = k_means(X_input, K,centroid)
 
  
  
  """
  
  

  
 
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

# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
   cAssign=assignClassification(centroids,X_input)
    for i in range(numRows):
     for j in range(numCol):
       #meancluster[cAssign[i],j]+= X_input[i,j]
       #k_counts[cAssign[i]]+=1
       #divide colums by the number of occurances k_count
    for k in range(K):
     for j in range(numCol):
       meancluster[k,j]= meancluster[k,j]/k_counts[k]
  centroids=meancluster
  print('hello')
  meancluster[:] = 0
  print('hello agian')
  k_counts.fill(0)
    """ 
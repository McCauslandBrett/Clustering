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
def getCentroid(centroids,X_input,clusterAssign ):

 numCentroids=centroids.shape[0]
 k_counts=[]
 for k in range(numCentroids):
   k_counts.append(0)
 numRows=X_input.shape[0]
 numCol=X_input.shape[1]
 print('clusterAssign',clusterAssign)
 mc=centroids
 mc[:] = 0
 print('mc',mc)
 #talle column values and take k count
 for row in range(numRows):
   Xrow= list(X_input.iloc[row])
   print('Xrow',Xrow)
   Addr= clusterAssign[row]
   print('Addr',Addr)
   k_counts[Addr]+=1
   print('k_counts',k_counts)
   for col in range(numCol):
     Sum= Xrow[col]
     print('Sum',Sum)
     mc.iloc[Addr,col]+=Sum
     print('mc.iloc[Addr,col]',mc.iloc[Addr,col])
 for k in range(numCentroids):
   for col in range(numCol): 
     value=mc.iloc[k,col]
     #print('value',value)
     D=k_counts[k]
     #print('D',D)
    # print('should be:',(value/D))
     mc.iloc[k,col]= (value/D)
    # print('is:',mc.iloc[k,col])
 print('mc',mc)
 return mc
def k_means(X_input, K,centroids):
  
  max_iteration=3
  #this is a (data point x 1) vector 
  cAssign=[]
  
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
    cAssign=assignClassification(centroids,X_input)
    centroids=getCentroid(centroids,X_input,cAssign)
    print('centroids',centroids)
  return cAssign

def Gatherclasses(data,classranges):
 classes = data.iloc[:,4].values
 length = data.iloc[:,4].size
 classranges.append(0)
 flowertype=classes[0]

 for count in range(length):
  if(flowertype!=classes[count]):
    classranges.append(count)
    flowertype=classes[count]
 classranges.append(length)
 return
def computeSSE(centroids,df):
 df.sort_values("class", inplace=True)
 classranges=[]
 Gatherclasses(df,classranges)

 Sum=0
 for everycluster in range(len(classranges)-1):
    temp=0
    cluster=df.iloc[classranges[everycluster]:classranges[everycluster+1],:-1]
    numRows=cluster.shape[0]
    for row in range(numRows):
       dist= minkowskiDist(list(centroids.iloc[everycluster]),list(cluster.iloc[row]),2)
       temp+=dist
    Sum+=temp
 return Sum
  
#----------- Question 1: k-Means----------
# 1)
   #import the data set
 data=pd.read_csv('IrisDataSet.csv')
   #shuffle the data
 data = data.sample(frac=1).reset_index(drop=True)
 #feature matrix input
 X_input = data.iloc[:,:-1]    
 Y_input = data.iloc[ :, -1:]   
 
 df=data  
 K=3
 centroid=X_input.iloc[0:K,:]
 save=centroid
 cAssign = k_means(X_input, K,centroid)
 df.iloc[:,-1]=cAssign
 catch=computeSSE(centroid,df)
 
"""
 df.sort_values("class", inplace=True)
 classranges=[]
 Gatherclasses(df,classranges)
 cluster=df.iloc[classranges[0]:classranges[0+1],:-1]
"""  
  
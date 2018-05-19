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

#precondiotion:
#postcondition:
def assignClassification(df_centroids,df_Xinput):
 
 # assign each point to the set corresponding to the closest centroid
 K=centroids.shape[0]
 #print(centroids)
 cAssign=[]
 numRows=df_Xinput.shape[0]
 d=[]
 
 #create assignment vector of correct size
 #initialized at random
 for i in range(numRows):
   cAssign.append(i%K)

 for row in range(numRows):
   for k in range(K):
     dist= minkowskiDist(list(df_centroids.iloc[k]),list(df_Xinput.iloc[row]),2)
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
 
#precondiotion:
#postcondition:
def getCentroid(df_centroids,df_Xinput,list_clusterAssign):

 numCentroids=df_centroids.shape[0]
 k_counts=[]
 for k in range(numCentroids):
   k_counts.append(0)
 numRows=df_Xinput.shape[0]
 numCol=df_Xinput.shape[1]
 mc=df_centroids
 mc[:] = 0
 
 #talle column values and take k count
 for row in range(numRows):
   Xrow= list(df_Xinput.iloc[row])
   Addr= list_clusterAssign[row]
   k_counts[Addr]+=1
   for col in range(numCol):
     Sum= Xrow[col]
     mc.iloc[Addr,col]+=Sum
 for k in range(numCentroids):
   for col in range(numCol): 
     value=mc.iloc[k,col]
     D=k_counts[k]
     mc.iloc[k,col]= (value/D)
 return mc

#precondiotion: 
#   df_centoid has been initialized with k rows x feature
#   df_Xinput has been initialized with rows x feature
#   int_k has no purpose
#postcondition:
     #returns a list of cluster assignments where 
     #index maps to df_Xinput 
#   centroid in invertainly changes value
def k_means(df_Xinput, int_k,df_centroids):
  
  max_iteration=5
  #this is a (data point x 1) vector 
  list_clustAssign=[]
  
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
    list_clustAssign=assignClassification(df_centroids,df_Xinput)
    df_centroids=getCentroid(df_centroids,df_Xinput,list_clustAssign)
  return list_clustAssign

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

#precondition:
#postcondition:
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
 print('centroid',centroid)
 cAssign = k_means(X_input, K,centroid)
 #----------- Question 2: k-Means Evaluation----------
 data=pd.read_csv('IrisDataSet.csv')
 x=[]
 y=[]
 for k in range(1,5):
   data = data.sample(frac=1).reset_index(drop=True)
   df=data
   X_input = data.iloc[:,:-1]
   centroid=X_input.iloc[0:k,:]
   cAssign=k_means(X_input, k,centroid)
   df.iloc[:,-1]=cAssign
   catch=computeSSE(centroid,df)
   y.append(catch)
   x.append(k)
 plt.scatter(x,y)
 
     
"""
 df.sort_values("class", inplace=True)
 classranges=[]
 Gatherclasses(df,classranges)
 cluster=df.iloc[classranges[0]:classranges[0+1],:-1]
"""  
  
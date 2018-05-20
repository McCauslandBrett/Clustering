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

#precondiotion:
#postcondition:
def minkowskiDist(list_R,list_Rn,p):
  sum=0
  count=len(list_R)
  for i in range(count):
    sum = sum + pow((abs(list_R[i]-list_Rn[i])),p)
  fp= 1/p
  return pow(sum,fp)

#precondiotion:
#postcondition:
def assignClassification(df_centroids,df_Xinput):
 

 numCentroids=df_centroids.shape[0] 
 numRows=df_Xinput.shape[0]
 list_clusterAssignmets=[0] * numRows
 list_tuple_dist=[]


 for row in range(numRows):
   for k in range(numCentroids):
     dist= minkowskiDist(list(df_centroids.iloc[k]),list(df_Xinput.iloc[row]),2)
     list_tuple_dist.append([dist,k])
   list_tuple_dist.sort(key=operator.itemgetter(0))
   tuple_best=list_tuple_dist[0]
   list_clusterAssignmets[row]=tuple_best[1]
   list_tuple_dist.clear()
 return list_clusterAssignmets
 
 
#precondiotion:
#  list_clusterAssign: initialized with class assignments for df_Xinput
#  list_clusterAssign is used to map to correct centroid
#  df_Xinput: contains the feature matrix
#postcondition:
#   returns a dataframe df_meanCentroid with the new centroids
def getCentroid(df_centroids,df_Xinput,list_clusterAssign):

 numCentroids=df_centroids.shape[0]
 k_counts=[] #divide colums by the number of occurances each class
 for k in range(numCentroids):
   k_counts.append(0)
 numRows=df_Xinput.shape[0]
 numCol=df_Xinput.shape[1]
 df_meanCentroid=df_centroids.copy()
 df_meanCentroid[:] = 0 
 
 #talle column values and take k count
 for row in range(numRows):
   Xrow= list(df_Xinput.iloc[row])
   Addr= list_clusterAssign[row]
   k_counts[Addr]+=1
   for col in range(numCol):
     Sum= Xrow[col]
     df_meanCentroid.iloc[Addr,col]+=Sum
 for k in range(numCentroids):
   for col in range(numCol): 
     value=df_meanCentroid.iloc[k,col]
     D=k_counts[k]
     df_meanCentroid.iloc[k,col]= (value/D)
 return df_meanCentroid

#precondiotion: 
#   df_centoid has been initialized with k rows x feature
#   df_Xinput has been initialized with rows x feature
#   int_k has no purpose
#postcondition:
     #returns a list of cluster assignments where 
     #index maps to df_Xinput 
#   centroid in invertainly changes value
def k_means(df_Xinput, int_k,df_centroids):
  
  max_iteration=10
  #this is a (data point x 1) vector 
  list_clustAssign=[]
  
# Repeat until nothing is moved around, or some max iteration
  for iteration in range(max_iteration):
    list_clustAssign=assignClassification(df_centroids,df_Xinput)
    df_centroids=getCentroid(df_centroids,df_Xinput,list_clustAssign)
  return list_clustAssign

#precondition: 
#   df_data is sorted by column 4 and column 4 cotains class labels
#postcondition:
#   list_classranges has been cleared and filled with new ranges
def Gatherclasses(df_data,list_classranges):
 list_classranges.clear()
 classes = df_data.iloc[:,4].values
 length = df_data.iloc[:,4].size
 list_classranges.append(0)
 flowertype=classes[0]

 for count in range(length):
  if(flowertype!=classes[count]):
    list_classranges.append(count)
    flowertype=classes[count]
 list_classranges.append(length)
 return

#precondition:
#   df_centroids:  centrods
#   df: datatable with cluster assignments
#postcondition:
#   returns the SSE https://hlab.stanford.edu/brian/error_sum_of_squares.html
#   df has been sorted 
def computeSSE(df_centroids,df):
    
 df.sort_values("class", inplace=True)
 list_classranges=[]
 Gatherclasses(df,list_classranges)

 Sum=0
 for everycluster in range(len(list_classranges)-1):
    df_cluster=df.iloc[list_classranges[everycluster]:list_classranges[everycluster+1],:-1]
    numRows=df_cluster.shape[0]
    for row in range(numRows):
       dist= minkowskiDist(list(df_centroids.iloc[everycluster]),list(df_cluster.iloc[row]),2)
       Sum+=dist
 return Sum
def k_meansKneePlot(df_data):
 x=[]
 y=[]
 for k in range(1,11):
   df_data = df_data.sample(frac=1).reset_index(drop=True)
   df=df_data.copy()
   X_input = df_data.iloc[:,:-1].copy()
   centroid= X_input.iloc[0:k,:].copy()
   cAssign=k_means(X_input, k,centroid)
   df.iloc[:,-1]=cAssign
   catch=computeSSE(centroid,df)
   y.append(catch)
   x.append(k)
 plt.scatter(x,y)  
 return
#----------- Question 1: k-Means----------
# 1)
   #import the data set
 data=pd.read_csv('IrisDataSet.csv')
   #shuffle the data
 data = data.sample(frac=1).reset_index(drop=True)
 #feature matrix input
 X_input = data.iloc[:,:-1].copy()   
 Y_input = data.iloc[ :, -1:].copy()   
 
 df=data.copy()  
 K=3
 centroid=X_input.iloc[0:K,:].copy()
 save=centroid
 print('centroid',centroid)
 cAssign = k_means(X_input, K,centroid)
 #----------- Question 2: k-Means Evaluation----------
 data=pd.read_csv('IrisDataSet.csv')
 x=[]
 y=[]
 for k in range(1,11):
   data = data.sample(frac=1).reset_index(drop=True)
   df=data.copy()
   X_input = data.iloc[:,:-1].copy()
   centroid=X_input.iloc[0:k,:].copy()
   cAssign=k_means(X_input, k,centroid)
   df.iloc[:,-1]=cAssign
   catch=computeSSE(centroid,df)
   y.append(catch)
   x.append(k)
 #plt.scatter(x,y)
 # multiple line plot
 plt.plot( x,y, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=1)
 #plt.plot( 'x', 'y2', x,y, marker='', color='olive', linewidth=2)
 #plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
 #plt.legend()

 y.clear()
 x.clear()
 print(df.iloc[0])
     
"""
 df.sort_values("class", inplace=True)
 classranges=[]
 Gatherclasses(df,classranges)
 cluster=df.iloc[classranges[0]:classranges[0+1],:-1]
"""  
  
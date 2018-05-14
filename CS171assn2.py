#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:43:48 2018

@author: Brettmccausland
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#precondition: data is the entire table data set,classes[],classranges[]
#postcondition: variables contain; classes(class column)
#classranges (beginning and ending of where  class rows end)
def Gatherclasses(data,classes,classranges):
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

def GenerateAllscatter(data,classes,classranges):
    classes = data.iloc[:,4].values  
    headers=list(data)
    #print(classes)
    headers.pop()
    
    numcol=len(headers)
    numclass=(len(classranges)-1)
    print(numclass)
    for k in range(numclass):
      for i in range(numcol):
        for j in range(i+1,numcol):
            x=data.iloc[classranges[k]:classranges[k+1],i].values
            y=data.iloc[classranges[k]:classranges[k+1],j].values
            plt.scatter(x,y)
            plt.xlabel(headers[i])
            plt.ylabel(headers[j])
            plt.title(headers[i]+' & '+headers[j])
            print('class flower='+classes[classranges[k]])
            print('attribute pair='+headers[i]+','+headers[j])
            plt.savefig(classes[classranges[k]]+' '+headers[i]+' '+headers[j])
            plt.close()
    return
#----------- Question 1: Feature distribution ----------
# 1)
   #import the data set
data=pd.read_csv('IrisDataSet.csv')
headers=list(data) # get every column(attribute) title
headers.pop() #remove the class column
classes=[]
classranges=[]
Gatherclasses(data,classes,classranges)
GenerateAllscatter(data,classes,classranges)



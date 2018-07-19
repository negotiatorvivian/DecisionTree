# -*- coding:utf-8 -*-
import pandas as pd 
import numpy as np 
import time
import graphviz
from sklearn import tree
from common import *

def read_file(filepath):
    file = pd.read_csv(filepath, names = columns)
    if file is None or False:
        return False
    return file

def Pretreatment(dataSet):
    length = len(dataSet)
    trainSetLen = int(length * 0.8)
    dataSet = dataSet.sample(frac = 1, random_state = 1)
    trainSet = dataSet.head(trainSetLen)
    testSet = dataSet.tail(length - trainSetLen)
    return trainSet, testSet

if __name__ == '__main__':    
    starttime = time.time()
    dataframe = read_file('/Users/zhangziwei/Downloads/testdata/zoo/zoo.csv')
    df, df1 = Pretreatment(dataframe)
    Y = np.array(df.type, dtype = int)
    df.drop(['animal_name', 'type'], axis = 1, inplace = True)
    dataSet = np.array(df)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(dataSet, Y)
    endtime = time.time()
    print(endtime - starttime)
    dot_data = tree.export_graphviz(clf , out_file = None , 
                         feature_names = df.columns,   
                         # class_names = df.target_names,   
                         filled = True,rounded = True,   
                         special_characters = True) 
    graph = graphviz.Source(dot_data)
    graph.render("iris")


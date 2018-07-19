# -*- coding:utf-8 -*-
from math import log
import pandas as pd 
import numpy as np 
from common import *
import time
from time import ctime
# from PlotTree import *

def countFunc(func):
    num = [0]
    def timef(*s,**gs):
        num[0] += 1
        print("执行次数",num[0])
        return func(*s,**gs)
    return timef

        
def read_file(filepath):
    file = pd.read_csv(filepath, names = columns)
    if file is None or False:
        return False
    file.drop('animal_name', axis = 1, inplace = True)
    return file

@countFunc
def calcalcShannonEnt(dataSet, types):
    shannonEnt = 0
    length = len(dataSet)
    for t in types:
        childSet = dataSet.loc[dataSet['type'] == t]         # 该子类中包含多少个类标签
        childSetLen = len(childSet)
        prob = float(childSetLen) / length
        shannonEnt -= prob * log(prob, 2) 
    return shannonEnt


def ChooseBestFeature(dataSet):
    featureNum = dataSet.shape[1]
    labels = set(dataSet.type)
    EntropyS = calcalcShannonEnt(dataSet, labels)
    IG = {}
    for i in range(featureNum - 1):
        EntropyST = 0.0
        column = dataSet.iloc[:, i]
        params = set(column)
        temp = 0
        for param in params:
            splitDataSet1 = dataSet.loc[dataSet.iloc[:, i] == param]         #该列有几种取值
            types = set(splitDataSet1.type)
            res = calcalcShannonEnt(splitDataSet1, types)
            prob = float(len(splitDataSet1)) / len(dataSet)
            EntropyST += prob * res 
        IG[i] = EntropyS - EntropyST
    IG = sorted(IG.items(), key = lambda x:x[1], reverse = True)
    if len(IG) > 0:
        index = IG[0][0]
        if index >= 0:
            return index
    return None
    


def splitData(dataSet, axis):
    index = dataSet.keys()[axis]
    values = set(dataSet.iloc[:, axis])
    childSet = {}
    for var in values:
        childSet[var] = dataSet.loc[dataSet.iloc[ :, axis] == var]
        childSet[var].drop([index], axis = 1)
    return childSet


def createDecisionTree(dataSet):
    index = ChooseBestFeature(dataSet)
    if index is not None:
        bestFeatLabel = dataSet.keys()[index] 
        myTree = {bestFeatLabel : {}} 
        dataSets = splitData(dataSet, index)
        for i in dataSets:
            if len(dataSets[i]) > 0:
                types = set(dataSets[i].type)
                if len(types) == 1:
                    myTree[bestFeatLabel][i] = types.pop()
                    continue
                myTree[bestFeatLabel][i] = createDecisionTree(dataSets[i])

        return myTree


def Pretreatment(dataSet):
    length = len(dataSet)
    trainSetLen = int(length * 0.8)
    dataSet = dataSet.sample(frac = 1, random_state = 1)
    trainSet = dataSet.head(trainSetLen)
    testSet = dataSet.tail(length - trainSetLen)
    return trainSet, testSet


def test_model(tree, dataSet):
    wrongSample = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == 'dict':
            temp = dataSet.loc[dataSet[first_key] == key]
            if len(temp) == 0:
                continue
            wrongSample += test_model(next_dict[key], temp)
        else:
            temp = dataSet.loc[dataSet[first_key] == key]
            wrongSample += len(temp) - len(temp.loc[temp.type == next_dict[key]])
            dataSet = dataSet.loc[dataSet[first_key] != key]
            # print(first_key, key, len(temp), wrongSample, set(temp.type), '\n', temp.loc[temp.type == next_dict[key]])
    return wrongSample


if __name__ == '__main__':    
    starttime = time.time()
    # decisionTree = DesicionTree.DesicionTree()
    dataframe = read_file('/Users/zhangziwei/Downloads/testdata/zoo/zoo.csv')
    trainSet, testSet = Pretreatment(dataframe)
    tree= createDecisionTree(trainSet)
    print(tree)
    wrongSample = test_model(tree, testSet)
    print(wrongSample)
    endtime = time.time()
    print(endtime - starttime)
    # createPlot(tree)






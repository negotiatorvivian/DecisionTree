# -*- coding:utf-8 -*-
from math import log
import pandas as pd 
import numpy as np 
from common import *
import time
# from PlotTree import *

ClassDict = {}
def read_file(filepath):
    file = pd.read_csv(filepath, names = columns)
    if file is None or False:
        return False
    file.drop('animal_name', axis = 1, inplace = True)
    return file


def calcalcShannonEnt(dataSet, types, i):
    gini = 1
    length = len(dataSet)

    for t in types:
        childSetLen = len(dataSet.loc[dataSet.type == t])
        prob = float(childSetLen)/length
        gini -= prob * prob
    # print(i, columns[i + 1], gini)
    return gini


def ChooseBestFeature(dataSet):
    featureNum = dataSet.shape[1]
    labels = set(dataSet.type)
    # EntropyS = calcalcShannonEnt(dataSet, labels)
    IG = {}
    for i in range(featureNum - 1):
        EntropyST = 0.0
        column = dataSet.iloc[:, i]
        # column = dataSet.type
        params = set(column)
        temp = 0
        if len(params) > 1:
            ginis = {}

            for param in params:
                gini = 0

                splitDataSet1 = dataSet.loc[dataSet.iloc[:, i] == param]         #该列有几种取值
                types = set(splitDataSet1.type)
                res = calcalcShannonEnt(splitDataSet1, types, i)
                prob = float(len(splitDataSet1)) / len(dataSet)
                gini += prob * res
                splitDataSet2 = dataSet.loc[dataSet.iloc[:, i] != param]         #该列有几种取值
                types = set(splitDataSet2.type)
                res = calcalcShannonEnt(splitDataSet2, types, i)
                prob = float(len(splitDataSet2)) / len(dataSet)
                gini += prob * res
                ginis[param] = gini
            ginis = sorted(ginis.items(), key = lambda x:x[1], reverse = False)
            ClassDict[i] = ginis[0][0]
            # IG[i] = [ginis[0][1], ginis[0][0]]
            IG[i] = ginis[0][1]
            
        else:
            for param in params:
                splitDataSet1 = dataSet.loc[dataSet.iloc[:, i] == param]         #该列有几种取值
                types = set(splitDataSet1.type)
                res = calcalcShannonEnt(splitDataSet1, types, i)
                prob = float(len(splitDataSet1)) / len(dataSet)
                EntropyST += prob * res
            # print(i, columns[i + 1], params, EntropyST) 
            IG[i] = EntropyST
    IG = sorted(IG.items(), key = lambda x:x[1], reverse = False)

    if len(IG) > 0:
        # print(IG)
        index = IG[0][0]
        if index >= 0:
            if index in ClassDict.keys():
                return index, ClassDict[index]
            return index, None
    return None
    


def splitData(dataSet, axis, value):
    index = dataSet.keys()[axis]
    # values = set(dataSet.iloc[:, axis])
    childSet = {}
    # for var in values:
    if value is None:
        value = set(dataSet.iloc[ :, axis]).pop()

    childSet[value] = dataSet.loc[dataSet.iloc[ :, axis] == value]
    childSet[value].drop([index], axis = 1)
    childSet['!' + str(value)] = dataSet.loc[dataSet.iloc[ :, axis] != value]
    childSet['!' + str(value)].drop([index], axis = 1)
    return childSet


def createDecisionTree(dataSet):
    index, var = ChooseBestFeature(dataSet)
    if index is not None:
        bestFeatLabel = dataSet.keys()[index] 
        myTree = {bestFeatLabel : {}} 
        dataSets = splitData(dataSet, index, var)
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
    dataframe = read_file('/Users/zhangziwei/Downloads/testdata/zoo/zoo.csv')
    trainSet, testSet = Pretreatment(dataframe)
    tree = createDecisionTree(trainSet)
    print(tree)
    endtime = time.time()
    print(endtime - starttime)
    # createPlot(tree)






#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# encapsulate into well defined module
class module:
    def __init__(self):
            self.data = None
            self.result = None
            self.minNum = None
            self.maxLevel = None
            self.tree = None
            
    # let decision tree grow for one layer
    def updateTree(self,tree):
        if tree.isLeaf:
            tree.grow(self.minNum)
        else:
            self.updateTree(tree.left)
            self.updateTree(tree.right)
            
    # pass dataset, results(prices), minleaf, maxdepth
    def train(self,data,result,minNum = 1,maxLevel = 10):
        self.data = data
        self.result = result
        self.minNum = minNum
        self.maxLevel = maxLevel
        self.tree = RegressionTree(range(data.shape[0]), data, result)
        
        # maxlevel determine for how deep a decision tree grows
        for i in range(maxLevel):
            self.updateTree(self.tree)

    # predict housing price using trained module
    def predict(self,data):
        result = []
        if(self.tree==None):
            return None
        for index, row in data.iterrows():
            result.append(self.singlePredict(row))
        return result
    
    # predict a single case    
    def singlePredict(self,data):
        if(self.tree==None):
            return None
        tree = self.tree
        while(True):
            if(tree.isLeaf):
                return tree.output
            if(data[tree.parameter]<tree.divide):
                tree = tree.left
            else:
                tree = tree.right

# define regression tree node
class RegressionTree:
    def __init__(self,sequence,data,result):
        self.data = data
        self.result = result
        self.isLeaf = True
        self.left =None
        self.right = None
        self.output = np.mean(result.iloc[sequence])
        self.sequence = sequence
        self.parameter = None
        self.divide = None

    # grow from current node
    def grow(self, minnum):
        if (len(self.sequence)<= minnum):
            return
        parameter,divide,err = bestdivide(self.data, self.result, self.sequence)
        left = []
        right =[]
        for i in self.sequence:
            if(self.data.iloc[i,parameter]<divide):
                left.append(i)
            else:
                right.append(i)
        self.parameter = parameter
        self.divide = divide
        self.isLeaf = False
        self.left = RegressionTree(left, self.data, self.result)
        self.right = RegressionTree(right, self.data, self.result)
            
# calculate sse according to parameter and divide point
def squaErr(data, result, sequence, parameter, divide):
    # select left and right subsequence
    left = []
    right = []
    for i in sequence:
        if data.iloc[i,parameter]<divide:
            left.append(i)
        else:
            right.append(i)
            
    # calculate SSEs for both subset
    c1 = np.mean(result.iloc[left])
    err1 = np.sum((result.iloc[left]-c1)**2)

    c2 = np.mean(result.iloc[right])
    err2 = np.sum((result.iloc[right]-c2)**2)
    # return sse
    return err1+err2


# select best dividing feature and its divide point
def bestdivide(data,result,sequence):
    min_para = -1
    min_divide = 0
    err = float("inf")
    # loop all features
    for para in range(data.shape[1]):
        # use mean as divide point
        mean = np.mean(data.iloc[sequence, para])
        errNew = squaErr(data,result,sequence,para,mean)
        if errNew < err:
            err = errNew
            min_para = para
            min_divide = mean
    return min_para, min_divide, err


# calculate sse to evaluate results    
def err(result, y):
    err = (result-y)**2
    return err.sum()


def normalize(dataset, feature_names):
    mins = []
    maxs = []
    for feature_name in feature_names[0:-1]:
        min_value = dataset[feature_name].min()
        max_value = dataset[feature_name].max()
        mins.append(min_value)
        maxs.append(max_value)
    for i, feature_name in enumerate(feature_names[0:-1]):
        dataset.loc[:, (feature_name)] = [(x - mins[i])/(maxs[i] - mins[i]) for x in dataset.loc[:,(feature_name)]]
    return dataset


# Housing dataset
feature_names1 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('Assignment1/housing.csv', header = None, names = feature_names1)
prob_name1 = 'Housing'
# ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.60, 0.70, 0.80, 0.90]
ratios = [0.05, 0.10, 0.15]
dataset = normalize(dataset, feature_names1)

length = len(dataset)
kf = KFold(n_splits=10, shuffle=True)
    
# find best ratio
best_ratio = 0
best_sse = 0
    
result_table = {}
result_table['average_sse'] = {}
result_table['std_deviation'] = {}
for ratio in ratios:
    threshold = round(ratio * length)
    sse_list = []
    tree = module()
    for train_index, test_index in kf.split(dataset):
        Xtrain = dataset.iloc[train_index, 0:-1]
        Xtest = dataset.iloc[test_index, 0:-1]
        Ytrain = dataset.iloc[train_index, -1]
        Ytest = dataset.iloc[test_index, -1]
        tree.train(Xtrain,Ytrain,threshold,10)
        Ypredict = tree.predict(Xtest)
        sse = err(Ypredict, Ytest)
        sse_list.append(sse)
    # end inner for
    avg_sse = np.mean(sse_list)
    if best_sse < avg_sse:
        best_sse = avg_sse
        best_ratio = ratio
    result_table['average_sse'][ratio] = np.mean(sse_list)
    result_table['std_deviation'][ratio] = np.std(sse_list, ddof=1)
# end outer for
result_table_df = pd.DataFrame(result_table, index = ratios, columns = ['average_sse', 'std_deviation'])
print(result_table_df)


# In[ ]:





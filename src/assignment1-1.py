#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from pandas import DataFrame
from math import log2
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

def numeric_2_categorical(data_set):
    """
    transform numeric data_set to a binary categorical data_set
    split by medians
    """
    num_feature = len(data_set.columns) - 1 # num of features
    new_data_set_dict = {}
    for feature_idx in range(num_feature):
        feature_val_list = data_set.iloc[:, feature_idx]
        median = feature_val_list.median()
        category_val_list = []
        for i, val in enumerate(feature_val_list):
            if val > median:
                category_val_list.append('>' + str(round(median, 6)))
            else:
                category_val_list.append('<=' + str(round(median, 6)))
        new_data_set_dict[data_set.columns[feature_idx]] = category_val_list  
    new_data_set_dict[data_set.columns[-1]] = data_set.iloc[:, -1]
    new_data_set = pd.DataFrame(new_data_set_dict, columns = data_set.columns)
    return new_data_set

def calculate_entropy(data_set):
    """Calculate entropy by data set label.
       formula: H(X) = -3/8*log(3/8, 2) - -5/8*log(5/8, 2)"""
    data_len = data_set.shape[0]
    entropy = 0
    for size in data_set.groupby(data_set.iloc[:, -1]).size():
        p_label = size/data_len
        entropy -= p_label * log2(p_label)
    return entropy


def get_best_feature(data_set):
    """Get the best feature by infoGain.
       formula: InfoGain(X, Y) = H(X) - H(X|Y)
                H(X|Y) = sum(P(X) * H(Yx))"""
    best_feature = -1
    base_entropy = calculate_entropy(data_set)
    best_info_gain = -1
    len_data = data_set.shape[0]
    for i in range(data_set.shape[1] - 1):
        new_entropy = 0
        for _, group in data_set.groupby(data_set.iloc[:, i]):
            p_label = group.shape[0]/len_data
            new_entropy += p_label * calculate_entropy(group)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_feature = i
            best_info_gain = info_gain
    return best_feature


def majority_cnt(class_list):
    """When only class label, return the max label."""
    majority_class = class_list.groupby(
        class_list.iloc[:, -1]).size().sort_values().index[-1]
    return majority_class


def create_tree(data_set, labels, threshold):
    """data_set: DataFrame"""
    class_list = data_set.values[:, -1]
    class_list_set = set(class_list)
    if len(class_list_set) == 1:
        return list(class_list)[0]
    if len(data_set.values[0]) == 1 or len(class_list) <= threshold:
        return majority_cnt(data_set)
    best_feature = get_best_feature(data_set)
    best_feature_label = labels[best_feature]
    sub_labels = labels.copy()
    del sub_labels[best_feature]
    my_tree = {best_feature_label: {}}
    for name, group in data_set.groupby(data_set.iloc[:, best_feature]):
        group.drop(columns=[best_feature_label], axis=1, inplace=True)
        my_tree[best_feature_label][name] = create_tree(group, sub_labels, threshold)
    return my_tree


def classify(test_data, my_tree, feat_labels):
    if not test_data:
        return 'Not found class.'
    first_str = list(my_tree.keys())[0]  # get first feature
    second_dict = my_tree[first_str]       
    feat_index = feat_labels.index(first_str) 
    for key in second_dict.keys():
        if test_data[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(test_data, second_dict[key], feat_labels)
            else:
                class_label = second_dict[key]
            return class_label   
        
def cal_accuracies(dataset, ratios, is_numeric):
    length = len(dataset)
    if is_numeric:
        dataset = numeric_2_categorical(dataset)
    kf = KFold(n_splits=10, shuffle=True)
    
    # find best ratio
    best_ratio = 0
    best_accuracy = 0
    
    result_table = {}
    result_table['average_accuracy'] = {}
    result_table['std_deviation'] = {}
    for ratio in ratios:
        threshold = round(ratio * length)
        accuracies = []
        for train_index, test_index in kf.split(dataset):
            train_data = dataset.iloc[train_index]
            feat_labels = list(dataset)
            tree = create_tree(train_data, feat_labels, threshold)
#             import json
#             print (json.dumps(tree, sort_keys=True, indent=2)) # 排序并且缩进两个字符输出
            feat_labels = list(dataset)
            test_data = dataset.iloc[test_index]
            err = 0
            for index, test_vec in test_data.iterrows():
                actual_class = test_vec[-1]
                test_vec = test_vec[0:-1].tolist()
                predict_class = classify(test_vec, tree, feat_labels)
                if predict_class != actual_class:
                    err += 1
            score = 1 - err/len(test_data)
            accuracies.append(score)
        # end for
        avg_accuracy = np.mean(accuracies)
        if best_accuracy < avg_accuracy:
            best_accuracy = avg_accuracy
            best_ratio = ratio
        result_table['average_accuracy'][ratio] = np.mean(accuracies)
        result_table['std_deviation'][ratio] = np.std(accuracies, ddof=1)
    result_table_df = pd.DataFrame(result_table, index = ratios, columns = ['average_accuracy', 'std_deviation'])
    print(result_table_df)
    
    return best_ratio

def generate_reports(best_ratio, dataset, is_numeric):
    length = len(dataset)
    threshold = round(best_ratio * length)
    if is_numeric:
        dataset = numeric_2_categorical(dataset)
    kf = KFold(n_splits=10, shuffle=True)
    class_names = np.unique(dataset["class"])
    # if class_names is [0, 1], transfer it to strs
    class_names = [ str(x) for x in class_names ]

    for train_index, test_index in kf.split(dataset):
        train_data = dataset.iloc[train_index]
        feat_labels = list(dataset)
        tree = create_tree(train_data, feat_labels, threshold)
        feat_labels = list(dataset)
        Ypredict = []
        test_data = dataset.iloc[test_index]
        for index, test_vec in test_data.iterrows():
            test_vec = test_vec[0:-1].tolist()
            predict_class = classify(test_vec, tree, feat_labels)
            Ypredict.append(predict_class)
        Ytest = test_data.iloc[:, -1]
    
        # transfer [0, 1] to strs
        Ypredict = [ str(x) for x in Ypredict ]
        Ytest = [ str(x) for x in Ytest ]

        matrix = confusion_matrix(Ytest, Ypredict, labels=class_names)
        print(matrix)
        report = classification_report(Ytest, Ypredict, target_names=class_names)
        print(report)
        break


# In[4]:


# Iris dataset
feature_names1 = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset1 = pd.read_csv('Assignment1/iris.csv', header = None, names = feature_names1)
prob_name1 = 'Iris'
ratios1 = [0.05, 0.10, 0.15, 0.20]

dataset1 = normalize(dataset1, feature_names1)
best_ratio = cal_accuracies(dataset1, ratios1, True)
print()
generate_reports(best_ratio, dataset1, True)


# In[7]:


# Spambase dataset
feature_names2 = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 
                  'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 
                  'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
                  'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
                  'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 
                  'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 
                  'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
                  'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
                  'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
                  'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 
                  'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
                  'capital_run_length_longest', 'capital_run_length_total', 'class']
dataset2 = pd.read_csv('Assignment1/spambase.csv', header = None, names = feature_names2)
prob_name2 = 'Spambase'
ratios2 = [0.05, 0.10, 0.15, 0.20, 0.25]

dataset2 = normalize(dataset2, feature_names2)
best_ratio = cal_accuracies(dataset2, ratios2, True)
print()
generate_reports(best_ratio, dataset2, True)


# In[ ]:





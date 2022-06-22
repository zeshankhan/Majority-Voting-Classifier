# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:03:41 2022

@author: ZESHAN KAHN
"""
from data_reading import train_test_building

dataset="me2018"#e2017,me2018,icpr2021
features="lire"#lbp,lire


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
data_path="/kaggle/input/kvasirv1/"
if(dataset=="me2017"):
    data_path="/kaggle/input/kvasirv1/"
if(dataset=="me2018"):
    data_path="/kaggle/input/kvasirv2/"

result_path="/kaggle/working/"

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

#clf = LogisticRegression(random_state=0,solver='liblinear')


clf1 = LogisticRegression(random_state=0,solver='liblinear')
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')


train_X, train_Y,test_X,test_Y=train_test_building(features, data_path, dataset)

clf.fit(train_X, train_Y)
pred_Y=clf.predict(test_X)

from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef
f=f1_score(test_Y, pred_Y, average='weighted')
acc=accuracy_score(test_Y, pred_Y)
mcc=matthews_corrcoef(test_Y, pred_Y)
print(f,acc,mcc)
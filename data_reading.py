# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:03:17 2022

@author: ZESHAN KAHN
"""
import numpy as np
import pandas as pd
def read_lbp(files):
    data=pd.DataFrame()
    for f in files:
        df=pd.read_csv(f).iloc[:,1:]
        df['full_name']=df['Yl']+df['filename'].astype(str)
        df.drop('Y',axis=1,inplace=True)
        df.drop('filename',axis=1,inplace=True)
        #print(df)
        if(len(data.columns)==0):
            data=df
        else:
            data=data.merge(df.iloc[:,1:], on='full_name',right_index=False)
            #print("Combined\t",data.shape)
    #print(data.iloc[5,:])
    Y=data['Yl']
    X=data
    X.set_index("full_name",inplace=True)
    X.drop('Yl',axis=1,inplace=True)
    X.drop('Yl_x',axis=1,inplace=True)
    X.drop('Yl_y',axis=1,inplace=True)
    X.astype(np.float32)
    return X,Y
def read_combine17(files):
    data=pd.DataFrame()
    for f in files:
        df=pd.read_csv(f,header=None)
        if(len(data.columns)==0):
            data=df
        else:
            data=data.merge(df.iloc[:,1:], on=1,right_index=False)
            print("Combined\t",data.shape)
    #print(data.iloc[5,:])
    X=data.iloc[:,2:].astype(np.float32)
    Y=data.iloc[:,0]
    return X,Y

def read_combine18(files):
    data=pd.DataFrame()
    for f in files:
        df=pd.read_csv(f).iloc[:,1:]
        df['full_name']=df['class1']+df['img']
        df.drop('img',axis=1,inplace=True)
        #print(df.shape)
        if(len(data.columns)==0):
            data=df
        else:
            data=data.merge(df.iloc[:,1:],how="inner", on='full_name',right_index=False)
            #print("Combined\t",data.shape)
    #print(data.iloc[5,:])
    Y=data['class1_y'].iloc[:,1]
    X=data
    X.set_index("full_name",inplace=True)
    X.drop('class1_x',axis=1,inplace=True)
    X.drop('class1_y',axis=1,inplace=True)
    X.astype(np.float32)
    #Y=data.iloc[:,-2]
    return X,Y

def train_test_building(features,data_path,dataset):
    train_X=None
    train_Y=None
    test_X=None
    test_X=None
    
    if(features=="lbp"):
        dev_files=[data_path+f for f in ["dev_lbp_1.csv","dev_lbp_2.csv","dev_lbp_3.csv","dev_lbp_4.csv","dev_lbp_5.csv"]]
        val_files=[f.replace("dev_","val_") for f in dev_files]
        train_X,train_Y=read_lbp(dev_files)
        test_X,test_Y=read_lbp(val_files)
        
    if(dataset=="me2017" and features=="lire"):
        dev_files=[data_path+f for f in ["dev_edge_histogram.csv","dev_tamura.csv","dev_jcd.csv","dev_color_layout.csv","dev_auto_color_correlation.csv","dev_phog.csv"]]
        val_files=[f.replace("dev_","val_") for f in dev_files]
        train_X,train_Y=read_combine17(dev_files)
        test_X,test_Y=read_combine17(val_files)
    if(dataset=="me2018" and features=="lire"):
        dev_files=[data_path+f for f in ["dev_EdgeHistogram.csv","dev_Tamura.csv","dev_JCD.csv","dev_ColorLayout.csv","dev_AutoColorCorrelogram.csv","dev_PHOG.csv"]]
        val_files=[f.replace("dev_","val_") for f in dev_files]
        train_X,train_Y=read_combine18(dev_files)
        test_X,test_Y=read_combine18(val_files)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    labelencoder.fit(train_Y.append(test_Y))
    train_Y=labelencoder.transform(train_Y)
    test_Y=labelencoder.transform(test_Y)
    #bridge_df['Bridge_Types_Cat'] = labelencoder.fit_transform(bridge_df['Bridge_Types'])
    
    train_X.astype(np.float64)
    train_Y.astype(np.float64)
    test_X.astype(np.float64)
    test_Y.astype(np.float64)
    return train_X, train_Y,test_X,test_Y
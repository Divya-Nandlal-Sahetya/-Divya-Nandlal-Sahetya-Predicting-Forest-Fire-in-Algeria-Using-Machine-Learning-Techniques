import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import random
import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def get_data_and_labels(df):
  labels = df["Classes"].to_frame()
  df = df.drop(columns="Classes", axis=1)
  return df,labels

def create_features(df, window, ignore_cols):
  stats = ['max_','min_','mean_','median_']
  for attribute in df.columns:
    if attribute in ignore_cols:
      continue

    df[stats[0]+attribute] = df.rolling(window, on="Date", axis=0, closed= "left").max().fillna(0)[attribute]
    df[stats[0]+attribute][1::2] = df.rolling(window+1, on="Date", axis=0, closed= "left").max().fillna(0)[attribute][1::2]
    
    df[stats[1]+attribute] = df[attribute].rolling(window, closed= "left").min().fillna(0)
    df[stats[1]+attribute][1::2] = df.rolling(window+1, on="Date", axis=0, closed= "left").min().fillna(0)[attribute][1::2]

    df[stats[2]+attribute] = df[attribute].rolling(window, closed= "left").mean().fillna(0)
    df[stats[2]+attribute][1::2] = df.rolling(window+1, on="Date", axis=0, closed= "left").mean().fillna(0)[attribute][1::2]

    df[stats[3]+attribute] = df[attribute].rolling(window, closed= "left").median().fillna(0)
    df[stats[3]+attribute][1::2] = df.rolling(window+1, on="Date", axis=0, closed= "left").median().fillna(0)[attribute][1::2]


def perform_data_standardization(data, method="StandardScaler"):
    # except date, take all columnsdata
    # cols = data.columns.difference(['Date'])
    cols = data.columns
    if method == "StandardScaler":
        scaler = StandardScaler(copy = True)
        # scaler.fit(data[cols])
        data[cols] = scaler.fit_transform(data[cols])
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler(copy = True)
        # scaler.fit(data[cols])
        data[cols] = scaler.fit_transform(data[cols])
    return data, scaler

## split of train and val data 
def split_data(df,labels, gap = 4, train_perc=0.8):
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_data, train_label = df[:train_size], labels[:train_size]
    val_data, val_label = df[train_size+gap:], labels[train_size+gap:]
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    return train_df,train_label,val_df,val_label

def PCA_Reduction(data, n_components=5, mode="train"):
  if mode == "train":
    pca = PCA(n_components=n_components)
    pca.fit(data)
    x_pc = pca.transform(data)
  elif mode == "test":
    x_pc = pca.transform(data)

  return x_pc
  
def get_correlated_features(data,threshold):
    col_corr = set()
    corr_matrix = data.corr()
    plt.rcParams['figure.figsize'] = (15,15)
    sns.heatmap(corr_matrix,annot=True,cmap='viridis',linewidths=.05)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


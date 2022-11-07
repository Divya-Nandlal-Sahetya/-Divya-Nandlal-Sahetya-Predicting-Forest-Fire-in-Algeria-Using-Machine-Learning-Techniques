import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import random
import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn import svm
from sklearn.metrics import RocCurveDisplay

def time_series_plot(data, save=True, save_loc="./images/png/time_series_plot.png", **kwargs):
    xcolumn = kwargs["xcolumn"]
    until = kwargs["until"]
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = data["Classes"]
    xval = list(range(len(data[xcolumn][:until])))
    fig, ax = plt.subplots(2, 5, figsize=(20,8))
    r = 0
    for idx, col in enumerate(data.columns):
        if col == xcolumn:
            continue
        if idx > 5:
            r = 1
        ax[r][idx%5].plot(xval, data[col][:until])
        if col != "Classes":
            ax[r][idx%5].scatter(xval, labels[:until]*data[col][:until].max(), color="orange")
        ax[r][idx%5].set_xlabel("Date")
        ax[r][idx%5].set_ylabel(col)
        ax[r][idx%5].set_title(f"{col} vs {xcolumn}")
        if col != "Classes":
            ax[r][idx%5].legend([f"{col}",f"Fire/No Fire"])
    fig.suptitle(f"Time Series plot for {until//2} days")
    # plt.tight_layout()
    # if save:
    print(f"save - {save}")
    plt.savefig(save_loc)
    # plt.savefig(save_loc)
    plt.show()

def preprocessing_plot(data,feature='Temperature',title = "Boxplot"):
    m = data[feature].mean()
    std = data[feature].std()
    print("Mean: " , m, " Standard Deviation: ",std )
    print("mean + std: " , m+std)
    print("mean - std: " , m-std)

    stud_bplt = data.boxplot(column = feature)
    stud_bplt.plot()
    plt.title(title)
    plt.show()

def pca_plot(explained_variance_ratio_,cols):
    # Plot
    plt.plot(range(0,len(explained_variance_ratio_)), explained_variance_ratio_*100)
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Explained Variance Ratio')
    plt.xticks(range(0,len(explained_variance_ratio_)), cols, rotation=60)
    plt.show()
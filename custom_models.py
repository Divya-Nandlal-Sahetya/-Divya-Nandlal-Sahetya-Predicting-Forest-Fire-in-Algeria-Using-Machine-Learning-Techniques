import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import random
import warnings
import seaborn as sns
from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
from sklearn.metrics import RocCurveDisplay
warnings.filterwarnings('ignore')

class TrivialSystem():
  def __init__(self, X, y, random_state=42):
    np.random.seed(random_state)
    self.random_state = random_state
    self.y = y
    self.prob_fire = np.mean(y)
    self.b = bernoulli(self.prob_fire)
    print(f"mean is {self.prob_fire}")    
  
  def fit(self, X, y, random_state=42):
    self.random_state = random_state
    self.y = y
    self.prob_fire = np.mean(y)
    self.b = bernoulli(self.prob_fire)
    print(f"mean is {self.prob_fire}")    
  
  def predict(self, data):
    print(data.shape)
    pred = [1 if np.random.rand() <= self.prob_fire else 0 for _ in range(len(data))]
    return pred
  
  def score(self, ypred, yact):
    correct = np.sum(ypred==yact.to_numpy())
    return correct / len(yact)
  
  def get_params(self, deep=True):
    return self.prob_fire

def distance(features, target, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((features-target)**2))

class NearestMeansClassifier():
    """Class that implements NearestMeans classifier
    """

    def __init__(self, training_data, training_labels):
      """class NearestMeansClassifier is an implementation for NearestMeans algorithm
      it accepts training data and labels to initialize and can be used as a classifier

      Args:
          training_data (ndarray): training_data using which classifier can be trained
          training_labels (ndarray): training_labels where index matches with the training_data
      """
      self.X = training_data
      self.y = training_labels
      self.means = []
      self.initialized_ = False

      #find all possible classes, assumes starts classes starts with 1
      self.C = np.unique(training_labels)

    def fit(self):
      """Trains the NearestMeans classifier

      Returns:
          means vector: means vector of shape no_of_classes x no_of_features
      """
      m_temp = []
      for c in self.C:
          x_c = self.X[np.where(self.y == c)]
          m_temp.append(np.mean(x_c, axis=0))
      self.means = np.asarray(m_temp)
      return self.means

    def predict(self, features):
      """Classifies features into one of the C classes

      Args:
          features (ndarray): feature_vector that needs to be classified

      Returns:
          prediction (ndarray): prediction for each row in features vector
      """
      if self.initialized_:
        print("Need to initialize NearestMeansClassifier prior to calling clf.predict", file=sys.stderr)
        sys.exit()

      pred = []
      for _ , feature in enumerate(features):
          err = []
          for c in self.C:
            err.append(distance(feature, self.means[c]))
          pred.append(np.argmin(err))
      return pred

    def score(self, ypred, labels):
      """Generates accuracy score

      Args:
          ypred (ndarray): predicted output
          labels (ndarray): labels to test the prediction vs target accuracy

      Returns:
          prediction_accuracy (float): #correct classification/ #total number of samples
      """
      # pred = self.predict(ypred)
      correctly_classified = (np.where(ypred == labels)[0])
      return len(correctly_classified)/float(len(labels))

    def error_rate(self, features, labels):
      """Generates error rate given as
      # of missclassification / #total number of samples

      Args:
          features (ndarray): feature vector to predict
          labels (ndarray): labels to test the prediction vs target error_rate

      Returns:
          prediction error_rate (float):  # of missclassification / #total number of samples
      """
      pred = self.predict(features)
      missed_clf = np.where(pred != labels)[0]
      return len(missed_clf)/float(len(labels))
    
      


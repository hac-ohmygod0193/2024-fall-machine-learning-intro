import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pandas as pd
# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

    @abstractmethod
    def predict_proba(self, X):
        # Abstract method predict the probability of the dataset X
        pass

# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=3, distance_metric='euclidean'): 
        self.k = k
        self.distance_metric = distance_metric
        print('Method:', self.distance_metric, 'K:', self.k)
    def _cal_distace(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            x1[i] = float(x1[i])
            x2[i] = float(x2[i])
            if(self.distance_metric == 'euclidean'):
                distance += (x1[i] - x2[i])**2
            elif(self.distance_metric == 'manhattan'):
                distance += abs(x1[i] - x2[i])
            elif(self.distance_metric == 'chebyshev'):
                distance = max(distance, abs(x1[i] - x2[i]))
        if(self.distance_metric == 'euclidean'):
            return np.sqrt(distance)
        else:
            return distance
        
    def fit(self, X, y): 
        if(isinstance(X, pd.DataFrame)):
            X = X.values
            y = y.values
        self.X_train = X
        self.y_train = y
        self.weight_0 = np.sum(y == 1) / len(y)
        self.weight_1 = np.sum(y == 0) / len(y)
    def predict(self, X):
        if(isinstance(X, pd.DataFrame)):
            X = X.values
        y_pred = [self._predict(x) for x in X] 
        return np.array(y_pred)
    
    def predict_proba(self, X):
        #TODO
        pass

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = self._cal_distace(x, x_train)
            distances.append(distance)
        k_indices = np.argsort(distances)[:self.k]
        label=0
        for i in range(len(k_indices)):
            if(self.y_train[k_indices[i]] == 0):
                label+=self.weight_0
                #label+=1
            else:
                label-=self.weight_1
                #label-=1
        if(label>0):
            return 0
        else:
            return 1
    def predict_score(self, pred, y):
        acc = accuracy_score(pred, y)
        f1 = f1_score(pred, y, zero_division=0)
        mcc = matthews_corrcoef(pred, y)
        scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
        
        return scoring
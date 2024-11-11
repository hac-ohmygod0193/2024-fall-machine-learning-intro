import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score


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
class TreeNode:
    def __init__(self, data, depth):
        self.data = data
        self.zero = np.sum(data == 0)
        self.one = np.sum(data == 1)

    def create_node(self, feature, threshold):
        self.attr = 'node'
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        
    def create_leaf(self):
        self.attr = 'leaf'
        if(self.zero > self.one):
            self.value = 0
        else:
            self.value = 1
    def predict_result(self, x):
        if self.left is None and self.right is None:
            return np.argmax([self.zero, self.one])
        if x[self.feature] < self.threshold:
            return self.left.predict_result(x)
        else:
            return self.right.predict_result(x)

class DecisionTreeClassifier:
    def __init__(self, max_depth=5,min_samples_leaf=1,method='entropy'):
        self.max_depth = max_depth
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        self.min_ig = 0.0
        self.method = method
    def fit(self, X, y):
        y = np.array(y['label'])
        self.tree = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        node = TreeNode(y, depth)
        if depth > self.max_depth or len(y) < self.min_samples_leaf or np.unique(y).shape[0] == 1:
            node.create_leaf()
            return node
        # find the best split
        feature, threshold = self.find_best_split(X, y)
        # split the dataset
        X_left, X_right, y_left, y_right = self.split_dataset(X, y, feature, threshold)
        
        # grow the tree
        node.create_node(feature, threshold)
        #print(len(y_left), len(y_right), len(y))
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):

        left_indices = X[feature_index] < threshold
        right_indices = X[feature_index] >= threshold

        X_left = X[left_indices]
        X_right = X[right_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]

        X_left = X_left.drop(columns=[feature_index])
        X_right = X_right.drop(columns=[feature_index])
        return X_left, X_right, y_left, y_right


    # Find the best split for the dataset
    def find_best_split(self, X, y):
        # X is dataframes
        # y is numpy array
        max_gain_ratio = float('-inf')
        best_feature, best_threshold = None, None
        for feature_index in X.columns:
            #print(f'Feature: {feature_index}')
            feature_values = X[feature_index].values
            for threshold in np.unique(feature_values):
                y_left,y_right = [], []
                #print("threshold", threshold, len(feature_values), len(y))
                for i in range(len(feature_values)):
                    if feature_values[i] < threshold:
                        y_left.append(y[i])
                    else:
                        y_right.append(y[i])
                gain_ratio = self._gain_ratio(y, y_left, y_right)
                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_feature = feature_index
                    best_threshold = threshold
                
        print(f'Best feature: {best_feature}, Best threshold: {best_threshold}, Max Gain Ratio: {max_gain_ratio}')
        return best_feature, best_threshold
    def entropy(self, y):
        # Calculate the entropy of the dataset
        return -np.sum([(np.sum(y == c) / len(y)) * np.log2(np.sum(y == c) / len(y)) for c in np.unique(y)])
    def gini(self, y):
        # Calculate the gini impurity of the dataset
        return 1 - np.sum([(np.sum(y == c) / len(y)) ** 2 for c in np.unique(y)])
    def _information_gain(self, y, y_left, y_right):
        # Calculate the information gain of a split
        if self.method=='entropy':
            ig = self.entropy(y) - (len(y_left) / len(y)) * self.entropy(y_left) - (len(y_right) / len(y)) * self.entropy(y_right)
        if self.method=='gini':
            ig = self.gini(y) - (len(y_left) / len(y)) * self.gini(y_left) - (len(y_right) / len(y)) * self.gini(y_right)
        return ig
    def _split_information(self,y_left, y_right):
        """Calculate Split Information."""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        p_left, p_right = n_left / n_total, n_right / n_total

        # Split information (Entropy of the split proportions)
        return -sum(p * np.log2(p) for p in [p_left, p_right] if p > 0)

    def _gain_ratio(self,y, y_left, y_right):
        """Calculate Gain Ratio."""
        ig = self._information_gain(y, y_left, y_right)
        si = self._split_information(y_left, y_right)
        
        # Avoid division by zero
        return ig / si if si != 0 else 0
    # prediction
    def predict_proba(self, X):
        raise NotImplementedError
    def predict_score(self, pred, y):
        acc = accuracy_score(pred, y)
        f1 = f1_score(pred, y, zero_division=0)
        mcc = matthews_corrcoef(pred, y)
        print(pred)
        #print(y)
        print('-' * 50)
        print(f'Acc: {acc:.5f}')
        print(f'F1 score: {f1:.5f}')
        print(f'MCC: {mcc:.5f}')
        scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
        print(f'Scoring: {scoring:.5f}')
        print('-' * 50)
    def predict(self, X):
        if self.tree is None:
            raise ValueError("Tree is empty")
        preds = []
        for i in range(len(X)):
            preds.append(self._predict_tree(X.iloc[i], self.tree))
        return preds

    def _predict_tree(self, X, node):
        if node.attr == 'leaf':
            return node.value
        if X[node.feature] < node.threshold:
            return self._predict_tree(X, node.left)
        else:
            return self._predict_tree(X, node.right)

    # print tree
    def print_tree(self):
        if self.tree is None:
            print("Tree is empty")
            return
        self._print_tree(self.tree)
    def _print_tree(self, node, depth=0, max_print_depth=2):
        
        if node is None:
            return
        if node.attr == 'node':
            print(f'{"  " * depth}[{node.feature}] [{node.zero} 0 / {node.one} 1]')
        else:
            print(f'{"  " * depth}[{node.value}]')
        if depth >= max_print_depth or node.attr == 'leaf':
            return
        if(node.left is not None and node.right is not None):
            print(f'{"  " * depth}Left:')
            self._print_tree(node.left, depth + 1)
            print(f'{"  " * depth}Right:')
            self._print_tree(node.right, depth + 1)
  
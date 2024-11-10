import numpy as np
from abc import ABC, abstractmethod
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
class TreeNode:
    def __init__(self, feature, threshold, depth, data, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.data = data
        self.zero = None
        self.one = None
        self.left = left
        self.right = right
    def create_leaf(self, data, depth):
        self.data = data
        self.zero = np.sum(data == 0)
        self.one = np.sum(data == 1)

class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
        self.min_samples_leaf = 1
        self.min_ig = 0.0
    def fit(self, X, y):
        y = np.array(y['label'])
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        
        if depth > self.max_depth or np.unique(y).shape[0] == 1:
            return None
        # find the best split
        feature, threshold, entropy = self.find_best_split(X, y)
        # split the dataset
        y_left, y_right, X = self.split_dataset(X, y, feature, threshold)
        

        # grow the tree
        node = TreeNode(feature, threshold, depth, y)
        print(len(X), len(y_left), len(y_right))
        node.left = self._grow_tree(X, y_left, depth + 1)
        node.right = self._grow_tree(X, y_right, depth + 1)
        return node

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):
        y_left, y_right = [], []
        for i in range(len(X[feature_index])):
            if X[feature_index][i] < threshold:
                y_left.append(y[i])
            else:
                y_right.append(y[i])
        X = X.drop(columns=[feature_index])
        return y_left, y_right, X


    # Find the best split for the dataset
    def find_best_split(self, X, y):
        # X is dataframes
        # y is numpy array
        min_entropy = float('inf')
        best_feature, best_threshold = None, None
        for feature_index in X.columns:
            print(f'Feature: {feature_index}')
            feature_values = X[feature_index].values
            for threshold in np.unique(feature_values):
                y_left,y_right = [], []
                print("threshold", threshold, len(feature_values), len(y))
                for i in range(len(feature_values)):
                    if feature_values[i] < threshold:
                        y_left.append(y[i])
                    else:
                        y_right.append(y[i])
                print(len(y_left), len(y_right))
                entropy_left = self.entropy(y_left)
                entropy_right = self.entropy(y_right)
                entropy_ = (len(y_left) / len(y)) * entropy_left + (len(y_right) / len(y)) * entropy_right
                if entropy_ < min_entropy:
                    min_entropy = entropy_
                    best_feature = feature_index
                    best_threshold = threshold
        print(f'Best feature: {best_feature}, Best threshold: {best_threshold}, Min entropy: {min_entropy}')
        return best_feature, best_threshold, min_entropy
    def entropy(self, y):
        # Calculate the entropy of the dataset
        return -np.sum([(np.sum(y == c) / len(y)) * np.log2(np.sum(y == c) / len(y)) for c in np.unique(y)])
    

    # prediction
    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        raise NotImplementedError
    

    # print tree
    def print_tree(self, max_print_depth=3):
        if self.tree is None:
            print("Tree is empty")
            return
        if self.tree.depth > max_print_depth:
            return
        self._print_tree(self.tree)
    def _print_tree(self, node, depth=0):
        if node is None:
            return
        print(f'{"  " * depth}[{node.feature}] [{node.zero} 0 / {node.one} 1]')
        print(f'{"  " * depth}Left:')
        self._print_tree(node.left, depth + 1)
        print(f'{"  " * depth}Right:')
        self._print_tree(node.right, depth + 1)


  
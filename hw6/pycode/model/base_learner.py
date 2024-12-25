import numpy as np
from abc import ABC, abstractmethod

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



# Logistic Regression Classifier
class LogisticRegressionClassifier(Classifier):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass

    

# ====== Activation funtion ====== #
class activation():
    def __init__(self):
        # TODO
        pass
    

# ====== Optimizer function ====== #
class optimizer():
    def __init__(self):
        # TODO
        pass


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



# MLP Classifier
class MLPClassifier(Classifier):
    def __init__(self, layers, activate_function, optimizer, learning_rate, n_epoch = 1000):
        """ TODO, Initialize your own MLP class """

        self.layers = layers
        self.activate_function = activate_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        
    def forwardPass(self, X):
        """ Forward pass of MLP """
        # TODO
        pass

    def backwardPass(self, y):
        """ Backward pass of MLP """
        # TODO
        pass

    def update(self):
        """ The update method to update parameters """
        # TODO
        pass
    
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        # TODO
        pass

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)



# Decision Tree Classifier
class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        raise NotImplementedError

    # Split dataset based on a feature and threshold
    def split_dataset(X, y, feature_index, threshold):
        raise NotImplementedError


    # Find the best split for the dataset
    def find_best_split(X, y):
        raise NotImplementedError

    def entropy(y):
        raise NotImplementedError
    

    # prediction
    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        raise NotImplementedError
    
    # print tree
    def print_tree(self, max_print_depth=3):
        raise NotImplementedError



# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=3, distance_metric='euclidean', p=3): 
        #TODO
        pass

    def fit(self, X, y): 
        #TODO
        pass

    def predict(self, X):
        y_pred = [self._predict(x) for x in X] 
        return np.array(y_pred)
    
    def predict_proba(self, X):
        #TODO
        pass

    def _predict(self, x):
        #TODO
        pass



# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.priors = {} # P(Class)
        self.likelihoods = {} # P(Feature|Class)

    def calculate_prior(self, y):
        # TODO
        pass

    def calculate_likelihood(self, X, y):
        # TODO
        pass

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass

    def predict_proba(self, X):
        # TODO
        pass



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


# Naive Bayes Classifier
class NaiveBayesClassifier:
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
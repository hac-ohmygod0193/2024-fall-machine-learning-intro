import numpy as np
from sklearn.model_selection import KFold
from base_learner import LogisticRegressionClassifier, MLPClassifier,   \
                DecisionTreeClassifier, KNearestNeighborClassifier,     \
                NaiveBayesClassifier
from tqdm import tqdm


class StackingClassifier:
    def __init__(self):
        """
        Stacking Classifier
        """
        pass

    def fit(self, X_train, y_train):
        """
        Fit the stacking model.
        :param X_train: Training data.
        :param y_train: Training labels.
        """
        pass

    def predict(self, X_test):
        """
        Predict using the stacking model.
        :param X_test: Test data.
        :return: Final predictions.
        """
        pass


import numpy as np
from sklearn.model_selection import KFold
from model.base_learner import LogisticRegressionClassifier, MLPClassifier,   \
                DecisionTreeClassifier, KNearestNeighborClassifier,     \
                NaiveBayesClassifier
from tqdm import tqdm


class StackingClassifier:
    def __init__(self):
        """
        Stacking Classifier
        :param base_learners: List of base learner models.
        :param meta_learner: Meta-learner model.
        :param n_splits: Number of splits for cross-validation.
        """
        # Define base learners
        base_learners = [
            ('logistic_regression', LogisticRegressionClassifier()),
            ('decision_tree', DecisionTreeClassifier()),
            ('knn', KNearestNeighborClassifier()),
            ('naive_bayes', NaiveBayesClassifier()),
            #('mlp', MLPClassifier())
        ]

        # Define meta-learner
        meta_learner = MLPClassifier()  # or MLPClassifier()
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_splits = 5

    def fit(self, X_train, y_train):
        """
        Fit the stacking model.
        :param X_train: Training data.
        :param y_train: Training labels.
        """
        self.base_learners_fitted = []
        n_samples, n_features = X_train.shape
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Store out-of-fold predictions
        oof_predictions = np.zeros((n_samples, len(self.base_learners)))

        for i, learner in enumerate(self.base_learners):
            learner_preds = np.zeros(n_samples)

            for train_idx, val_idx in tqdm(kf.split(X_train), desc=f"Training base learner {i+1}/{len(self.base_learners)}"):
                # Train on training fold
                print(f"\nTraining base learner {learner[0]}")
                learner_clone = learner[1]  # Create a new instance
                print(type(X_train[train_idx]))
                learner_clone.fit(X_train[train_idx], y_train[train_idx])
                
                # Predict on validation fold
                learner_preds[val_idx] = learner_clone.predict(X_train[val_idx])

            oof_predictions[:, i] = learner_preds
            self.base_learners_fitted.append(learner)

        # Train meta-learner on out-of-fold predictions
        self.meta_learner.fit(oof_predictions, y_train)

    def predict(self, X_test):
        """
        Predict using the stacking model.
        :param X_test: Test data.
        :return: Final predictions.
        """
        # Collect base learner predictions
        base_preds = np.column_stack([
            learner[1].predict(X_test) for learner in self.base_learners_fitted
        ])

        # Meta-learner predicts based on base learner outputs
        return self.meta_learner.predict(base_preds)

import numpy as np
from sklearn.model_selection import KFold
from model.base_learner import LogisticRegressionClassifier, MLPClassifier,   \
                DecisionTreeClassifier, KNearestNeighborClassifier,     \
                NaiveBayesClassifier
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

class StackingClassifier:
    def __init__(self, 
            base_learners=[
                #('logistic_regression', LogisticRegressionClassifier()),
                ('decision_tree', DecisionTreeClassifier()),
                ('knn', KNearestNeighborClassifier()),
                ('naive_bayes', NaiveBayesClassifier()),
                ('mlp', MLPClassifier())
            ],
            meta_learner = LogisticRegressionClassifier(),
        ):
        """
        Stacking Classifier
        :param base_learners: List of base learner models.
        :param meta_learner: Meta-learner model.
        :param n_splits: Number of splits for cross-validation.
        """

        #meta_learner = MLPClassifier()
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_splits = 5

    def fit(self, X_train, y_train):
        """
        Fit the stacking model.
        :param X_train: Training data (pandas DataFrame).
        :param y_train: Training labels (pandas Series).
        """
        if(isinstance(X_train, np.ndarray)):
            X_train = pd.DataFrame(X_train)
        if(isinstance(y_train, np.ndarray)):
            y_train = pd.DataFrame(y_train, columns=['label'])
        # Convert input to numpy arrays for processing
        X_train_np = X_train.values
        y_train_np = y_train.values

        self.base_learners_fitted = []
        n_samples, n_features = X_train_np.shape
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        # Store out-of-fold predictions
        oof_predictions = np.zeros((n_samples, len(self.base_learners)))

        for i, learner in enumerate(self.base_learners):
            learner_preds = np.zeros(n_samples)
            #print(f"Training base learner {i+1}/{len(self.base_learners)}")
            #print(f"Base learner: {learner[0]}")
            for train_idx, val_idx in tqdm(kf.split(X_train_np), desc=f"Training base learner {i+1}/{len(self.base_learners)}"):
                # Train on training fold
                learner_clone = learner[1]  # Create a new instance
                learner_clone.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                
                # Predict on validation fold
                learner_preds[val_idx] = learner_clone.predict(X_train.iloc[val_idx])

            oof_predictions[:, i] = learner_preds
            self.base_learners_fitted.append(learner)

        # Train meta-learner on out-of-fold predictions (convert to DataFrame for consistency)
        # Combine original features with out-of-fold predictions
        oof_predictions_df = pd.DataFrame(oof_predictions, columns=[f"BaseLearner_{i}" for i in range(len(self.base_learners))])
        self.meta_learner.fit(oof_predictions_df, y_train)

    def predict(self, X_test):
        """
        Predict using the stacking model.
        :param X_test: Test data (pandas DataFrame).
        :return: Final predictions.
        """
        # Collect base learner predictions
        base_preds = np.column_stack([
            learner[1].predict(X_test) for learner in self.base_learners_fitted
        ])
        #X_test_combined = X_test.join(pd.DataFrame(base_preds, columns=[f"BaseLearner_{i}" for i in range(len(self.base_learners))]))
        # Convert to DataFrame for meta-learner
        base_preds_df = pd.DataFrame(base_preds, columns=[f"BaseLearner_{i}" for i in range(len(self.base_learners))])
        base_preds_df.to_csv('base_preds.csv', index=False)
        # Meta-learner predicts based on base learner outputs
        return self.meta_learner.predict(base_preds_df)
    def predict_score(self, y_pred, y_true):
        """
        Calculate the scoring metric.
        :param y_pred: Predicted labels.
        :param y_true: True labels.
        :return: Scoring metric.
        """
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("F1:", f1_score(y_true, y_pred))
        print("MCC:", matthews_corrcoef(y_true, y_pred))
        return 0.3 * accuracy_score(y_true, y_pred) + 0.35 * f1_score(y_true, y_pred) + 0.35 * matthews_corrcoef(y_true, y_pred)
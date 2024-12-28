import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass

class NaiveBayesClassifier(Classifier):
    def __init__(self, mode='continuous'):
        self.mode = mode # 'continuous', 'discrete' or 'mixed'
        self.priors = {}  # P(Class)
        self.classes = None
        self.features_mean = {}  # For continuous features
        self.features_var = {}   # For continuous features
        self.categorical_prob = {}  # For categorical features
        self.continuous_features = None
        self.categorical_features = None
    def calculate_prior(self, y):
        """Calculate prior probabilities P(Class) for each class"""
        total_samples = len(y)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        return dict(zip(unique_classes, class_counts / total_samples))
    
    def calculate_continuous_likelihood(self, X_continuous, y):
        """Calculate likelihood parameters for continuous features"""
        features_mean = {}
        features_var = {}
        
        for c in self.classes:
            X_class = X_continuous[y.iloc[:, -1] == c]
            if not X_class.empty:  # If there are continuous features
                features_mean[c] = X_class.mean()
                features_var[c] = X_class.var()
            else:
                features_mean[c] = pd.Series([])
                features_var[c] = pd.Series([])
                
        return features_mean, features_var
    
    def calculate_categorical_likelihood(self, X_categorical, y):
        """Calculate likelihood probabilities for categorical features"""
        categorical_prob = {}
        #data =  pd.concat([X_categorical, y], axis=1)
        print("-----------------")
        #print(X_categorical[y.iloc[:, -1] == 1])
        for c in self.classes:
            categorical_prob[c] = {}
            X_class = X_categorical[y.iloc[:, -1] == c]
            #print(X_class)
            for col in X_categorical.columns:
                # Calculate probability of each category given the class
                value_counts = X_class[col].value_counts()
                # Add Laplace smoothing
                smoothed_counts = value_counts + 1
                probs = smoothed_counts / smoothed_counts.sum()
                categorical_prob[c][col] = probs.to_dict()
                
        return categorical_prob
    
    def gaussian_likelihood(self, x, mean, var):
        """Calculate Gaussian likelihood P(Feature|Class) for continuous features"""
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    def log_likelihood(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var
    def fit(self, X, y):
        """
        Fit the Naive Bayes model
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features
        y : array-like
            Target values
        """
        # Select features
        self.continuous_features = [col for col in X.columns if X[col].unique().shape[0] > 2]
        self.categorical_features = [col for col in X.columns if X[col].unique().shape[0] <=2]
        print(f"Continuous features: {self.continuous_features}")
        print(f"Categorical features: {self.categorical_features}")
        if(self.mode == 'continuous' and len(self.continuous_features) < 3):
            self.mode = 'mixed'
        # Split features
        X_continuous = X[self.continuous_features]
        X_categorical = X[self.categorical_features]
        
        self.classes = np.unique(y)
        self.priors = self.calculate_prior(y)
        #print(self.priors)
        # Calculate likelihood parameters for continuous features
        if(self.mode == 'continuous' or self.mode == 'mixed'):
            self.features_mean, self.features_var = self.calculate_continuous_likelihood(X_continuous, y)
        #print(self.features_mean)
        # Calculate likelihood probabilities for categorical features
        if(self.mode == 'discrete' or self.mode == 'mixed'):
            self.categorical_prob = self.calculate_categorical_likelihood(X_categorical, y)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features
        """
        # Split features using saved feature lists
        X_continuous = X[self.continuous_features]
        X_categorical = X[self.categorical_features]
        
        probas = np.zeros((len(X), len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Calculate prior probability
            class_prob = np.log(self.priors[c])
            
            # Calculate likelihood for continuous features
            if self.mode == 'continuous' or self.mode == 'mixed':
                for feature in self.continuous_features:
                    x = X_continuous[feature]
                    mean = self.features_mean[c][feature]
                    var = self.features_var[c][feature]
                    '''
                    class_prob += np.log(self.gaussian_likelihood(
                        X_continuous[feature],
                        self.features_mean[c][feature],
                        self.features_var[c][feature]
                    ))
                    '''
                    class_prob += self.log_likelihood(x, mean, var)
            if self.mode == 'discrete' or self.mode == 'mixed':

                # Calculate likelihood for categorical features
                for feature in self.categorical_features:
                    for idx in range(len(X)):
                        category = X_categorical.iloc[idx][feature]
                        # If category wasn't seen during training, use Laplace smoothing probability
                        prob = self.categorical_prob[c][feature].get(category, 1.0 / len(self.categorical_prob[c][feature]))
                        
                        class_prob += np.log(prob)
            
            probas[:, i] = class_prob
        log_prob_max = np.max(probas, axis=1, keepdims=True)
        probs = np.exp(probas - log_prob_max)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        # Convert log probabilities to normal probabilities
        #probas = np.exp(probas)
        # Normalize probabilities
        #probas = probas / np.sum(probas, axis=1)[:, np.newaxis]
        
        return probas
    
    def predict(self, X):
        """Predict class labels for samples"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    def predict_score(self, pred, y):
        acc = accuracy_score(pred, y)
        f1 = f1_score(pred, y, zero_division=0)
        mcc = matthews_corrcoef(pred, y)
        scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
        
        return scoring
    def categorical_analyze(self, X, threshold=0.3):
        import matplotlib.pyplot as plt
        # Calculate the correlation matrix
        corr = X[self.categorical_features].corr()
        count_high_corr = (corr.abs() > threshold).sum().sum() - len(corr)
        print(f"Count of high correlations: {count_high_corr}, len(corr): {len(corr)}")
        # Drop rows and columns with abs(correlation) > 0.3
        to_drop = [column for column in corr.columns if (corr[column].abs() > threshold).sum() > 5]
        print(f"Columns to drop: {to_drop}")
        print(f"Number of columns to drop: {len(to_drop)}")
        corr =  corr.drop(columns=to_drop).drop(index=to_drop)
        self.categorical_features = corr.columns
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap for Categorical Features")
        #plt.show()
    def continuous_analyze(self, X, threshold=0.3):
        import matplotlib.pyplot as plt

        # Calculate the correlation matrix
        corr = X[self.continuous_features].corr()
        # Calculate the count of correlations with absolute value > 0.2
        count_high_corr = (corr.abs() > threshold).sum().sum() - len(corr)
        print(f"Count of high correlations: {count_high_corr}, len(corr): {len(corr)}")
        # Drop rows and columns with abs(correlation) > 0.3
        to_drop = [column for column in corr.columns if (corr[column].abs() > threshold).sum() > 5]
        print(f"Columns to drop: {to_drop}")
        print(f"Number of columns to drop: {len(to_drop)}")
        corr =  corr.drop(columns=to_drop).drop(index=to_drop)# Create a heatmap
        self.continuous_features = corr.columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap for Continuous Features")
        #plt.show()
    def plot_continuous_distribution(self,X):
        # Plot the distribution for each continuous feature in one figure
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.continuous_features):
            plt.subplot(len(self.continuous_features) // 3 + 1, 3, i + 1)
            sns.histplot(X[feature], kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    def plot_categorical_distribution(self,X):
        # Plot the distribution for each categorical feature in one figure
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.categorical_features):
            plt.subplot(len(self.categorical_features) // 3 + 1, 3, i + 1)
            sns.countplot(data=X, x=feature)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
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
    def __init__(self, C=1.0, penalty='l2', lr=1e-2, iterations=600):
        super(LogisticRegressionClassifier, self).__init__()

        self.iterations = iterations
        self.lr= lr
        self.C = C
        self.penalty = penalty

    def sigmoid(self, x):
        """ The sigmoid function """
        return 1.0 / (1.0 + np.exp(-x))
    
    def linear(self, X):
        return np.dot(X, self.W) + self.b
    
    def SGD(self, dW, db):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def binaryCrossEntropy(self, pred, target):
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    def fit(self, X, y):
        m, n = X.shape # (m, n)
        self.W = np.random.randn(n) # (n,)
        # self.W = np.zeros(n)
        self.b = 0
        self.loss = []

        y = np.array(y).flatten()  # make sure y is one dimension

        for i in range(self.iterations):
            z = self.linear(X)
            y_hat = self.sigmoid(z)

            dW = 1/m * np.dot(X.T, (y_hat - y))
            db = 1/m * np.sum(y_hat - y)

            if self.penalty == 'l2':
                dW += (self.C / m) * self.W
            elif self.penalty == 'l1':
                dW += (self.C / m) * np.sign(self.W)

            # update weight and bias
            self.SGD(dW, db)

            # compute loss
            loss = self.binaryCrossEntropy(y_hat, y)
            self.loss.append(loss)
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_hat]
    
    def predict_proba(self, X):
        z = self.linear(X)
        return self.sigmoid(z)

    
    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(pred, y)
 
    
# ====== Activation funtion ====== #
class activation():
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        """ The sigmoid function """
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """ Derivative of the sigmoid function """
        sig = activation.sigmoid(x)
        return sig * (1.0 - sig)

    @staticmethod
    def relu(x):
        """ The relu function """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ Derivative of the relu function """
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        """ The tanh function """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """ Derivative of the tanh function """
        return 1.0 - np.tanh(x) ** 2
    
    

# ====== Optimizer function ====== #
class optimizer():
    def __init__(self):
        pass
    def SGD(W, b, dW, db,learning_rate):
        for i in range(len(W)):
            W[i] -= learning_rate * dW[i]
            b[i] -= learning_rate * db[i]
        return W, b
    def Adam(W, b, dW, db, learning_rate, m, v, t):
        t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for i in range(len(W)):
            m[i] = beta1 * m[i] + (1 - beta1) * dW[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (dW[i] ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            W[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            b[i] -= learning_rate * db[i]
        return W, b, m, v, t

    def Momentum(W, b, dW, db, learning_rate, velocity, beta=0.9):
        for i in range(len(W)):
            velocity[i] = beta * velocity[i] + (1 - beta) * dW[i]
            W[i] -= learning_rate * velocity[i]
            b[i] -= learning_rate * db[i]
        return W, b, velocity
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
    
  
class MLPClassifier(Classifier):
    def __init__(self, layers= [20,10] , activate_function=activation.sigmoid, activate_derivative=activation.sigmoid_derivative, optimizer=optimizer.Adam, learning_rate=0.05, n_epoch = 100000):
        self.hidden_layers = layers
        self.activate_function = activate_function
        self.activate_derivative = activate_derivative
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        
    def forwardPass(self, X, validate=False):
        """ Forward pass of MLP """
        if validate:
            a_list = [X]  # Use a separate list for validation activations
        else:
            self.a = [X]  # Keep the original activation list for training

        for i in range(len(self.W)):
            z = np.dot(a_list[-1], self.W[i]) + self.b[i] if validate else np.dot(self.a[-1], self.W[i]) + self.b[i]
            a = self.activate_function(z)

            if validate:
                a_list.append(a)  # Append to the validation list
            else:
                self.a.append(a)  # Append to the training list

        return a_list[-1] if validate else self.a[-1]

    def backwardPass(self, y):
        """ Backward pass of MLP """
        # Compute the gradients of the loss with respect to the weights and biases
        m = y.shape[0]
        dz = self.a[-1] - y.reshape(-1, 1)
        self.dW = []
        self.db = []
        for i in reversed(range(len(self.W))):
            dW = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.dW.insert(0, dW)
            self.db.insert(0, db)
            if i > 0:
                # compute the gradient of the loss with respect to the activations
                dz = np.dot(dz, self.W[i].T) * self.activate_derivative(self.a[i])
        
    def update(self):
        """ The update method to update parameters """
        # Update the weights and biases
        if self.optimizer == optimizer.Adam:
            if not hasattr(self, 'm'):
                self.m = [np.zeros_like(w) for w in self.W]
                self.v = [np.zeros_like(w) for w in self.W]
                self.t = 0
            self.W, self.b, self.m, self.v, self.t = self.optimizer(self.W, self.b, self.dW, self.db, self.learning_rate, self.m, self.v, self.t)
        elif self.optimizer == optimizer.SGD:
            self.W, self.b = self.optimizer(self.W, self.b, self.dW, self.db, self.learning_rate)
        elif self.optimizer == optimizer.Momentum:
            if not hasattr(self, 'velocity'):
                self.velocity = [np.zeros_like(w) for w in self.W]
            self.W, self.b, self.velocity = self.optimizer(self.W, self.b, self.dW, self.db, self.learning_rate, self.velocity)
    def binaryCrossEntropy(self, y_hat, y):
        # add an epsilon to avoid taking the log of zero
        epsilon = 1e-15  # Small constant to avoid log(0)
        # Clip predictions to avoid 0s and 1s
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Calculate loss
        m = y.shape[0]  # number of examples
        loss = -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        if(isinstance(X_train, pd.DataFrame)):
            X_train = X_train.values
            y_train = y_train.values
        data_len = X_train.shape[0]
        # Split the data into training and validation sets
        split = int(data_len * 0.8)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

        # Initialize weights and biases
        self.W = []
        self.b = []
        self.loss = []
        self.val_loss = []
        self.layers = [X_train.shape[1]] + self.hidden_layers + [1]
        y_train = np.array(y_train).flatten()  # make sure y is one dimension
        
        # Weight initialization should be based on the layer size
        intialize_method = ['he', 'xavier']
        initialization = intialize_method[1]
        
        if initialization == 'he':
            # He initialization
            for i in range(len(self.layers) - 1):
                self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i]))
                self.b.append(np.zeros((1, self.layers[i + 1])))
        else:
            # Xavier initialization
            for i in range(len(self.layers) - 1):
                self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(1.0 / self.layers[i]))
                self.b.append(np.zeros((1, self.layers[i + 1])))
        # Training loop
        for epoch in range(self.n_epoch):
            
            # Forward pass
            y_hat = self.forwardPass(X_train)
            # Compute loss (binary cross-entropy)
            loss = self.binaryCrossEntropy(y_hat, y_train)
            
            # Validation loss
            if(epoch%100==0):
                if X_val is not None and y_val is not None:
                    y_val_hat = self.forwardPass(X_val,True)
                    val_loss = self.binaryCrossEntropy(y_val_hat, y_val)
                    self.val_loss.append(val_loss)
                self.loss.append(loss)
            if(epoch%(self.n_epoch/5)==0):
                self.learning_rate*=0.9
            #if(epoch%(self.n_epoch/10)==0):
                #print(f'Epoch {epoch + 1}/{self.n_epoch}, Loss: {loss}, Val Loss: {val_loss if X_val is not None else "N/A"}')
                
            # Backward pass
            self.backwardPass(y_train)
            
            # Update weights and biases
            self.update()
            
            # Early stopping
            if len(self.val_loss) > 5 and all(self.val_loss[-i] > self.val_loss[-i-1] for i in range(1, 6)):
                print("Early stopping due to increase in validation loss at epoch = ", epoch)
                break
                

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)

# Decision Tree Classifier
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

class DecisionTreeClassifier:
    def __init__(self, max_depth=25, min_samples_leaf=5, method='gini'):
        self.max_depth = max_depth
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = 0.01
        self.method = method
    def fit_post_prune(self, X, y):
        #print("Post-pruning")
        total = len(X)
        train_size = int(total * 0.8)
        train_X, val_X = X[:train_size], X[train_size:]
        train_y, val_y = y[:train_size], y[train_size:]
        train_y = np.array(train_y['label'])
        self.tree = self._grow_tree(train_X, train_y)
        self.post_prune(val_X, val_y)

    def fit(self, X, y, post_prune=True):
        #print(f"Method: {self.method}, Max depth: {self.max_depth}, Min samples leaf: {self.min_samples_leaf}")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.DataFrame(y, columns=['label'])
        if post_prune:
            self.fit_post_prune(X, y)
        # split the training data into training and validation data
        else:
            y = np.array(y['label'])
            self.tree = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'F{i+1}' for i in range(X.shape[1])])
        node = TreeNode(y, depth)
        if depth > self.max_depth or len(y) < self.min_samples_leaf or np.unique(y).shape[0] == 1:
            node.create_leaf()
            return node
        # find the best split
        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            node.create_leaf()
            return node
        # split the dataset
        X_left, X_right, y_left, y_right = self.split_dataset(X, y, feature, threshold)
        
        # grow the tree
        node.create_node(feature, threshold)
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'F{i+1}' for i in range(X.shape[1])])
        left_indices = X[feature_index] < threshold
        right_indices = X[feature_index] >= threshold

        X_left = X[left_indices]
        X_right = X[right_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]
        
        return X_left, X_right, y_left, y_right


    # Find the best split for the dataset
    def find_best_split(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'F{i+1}' for i in range(X.shape[1])])
        max_gain_ratio = float('-inf')
        best_feature, best_threshold = None, None
        for feature_index in X.columns:
            feature_values = X[feature_index].values
            for threshold in np.unique(feature_values):
                y_left,y_right = [], []
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
        if max_gain_ratio < self.min_gain_ratio:
            return None, None
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
    def predict_score(self, pred, y):
        acc = accuracy_score(pred, y)
        f1 = f1_score(pred, y, zero_division=0)
        mcc = matthews_corrcoef(pred, y)
        scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
        
        return scoring
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'F{i+1}' for i in range(X.shape[1])])
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
    def print_tree(self, max_print_depth=3):
        if self.tree is None:
            print("Tree is empty")
            return
        self._print_tree(self.tree,max_print_depth=3)
    def _print_tree(self, node, depth=0, max_print_depth=3):
        max_print_depth-=1 # because start from 0
        if node is None:
            return
        if node.attr == 'node':
            print(f'{"  " * depth}[{node.feature}] [{node.zero} 0 / {node.one} 1] [Threshold = {node.threshold}]')
        else:
            print(f'{"  " * depth}[{node.value}]')
        if depth >= max_print_depth or node.attr == 'leaf':
            return
        if(node.left is not None and node.right is not None):
            print(f'{"  " * depth}Left:')
            self._print_tree(node.left, depth + 1)
            print(f'{"  " * depth}Right:')
            self._print_tree(node.right, depth + 1)
    # post-pruning
    def post_prune(self, validation_X, validation_y):
        if self.tree is None:
            raise ValueError("Tree is empty")
        self.prune_node(self.tree, validation_X, validation_y)
    def prune_node(self, node, validation_X, validation_y):
        if node.attr == 'leaf':
            return
        # Calculate current performance
        original_score = self.predict_score(self.predict(validation_X), validation_y)
        
        # Try making node a leaf
        temp_attr = node.attr
        node.create_leaf()
        
        # Calculate new performance
        new_score = self.predict_score(self.predict(validation_X), validation_y)
        
        # Revert if no improvement
        if new_score <= original_score:
            node.attr = temp_attr
        else:
            print(f'Pruned node: {node.feature} {node.threshold}')    
        # Continue pruning children if not leaf
        if node.attr != 'leaf':
            self.prune_node(node.left, validation_X, validation_y)
            self.prune_node(node.right, validation_X, validation_y)
  
# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=31, distance_metric='manhattan'): 
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


# Naive Bayes Classifier
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
        #print("-----------------")
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
        if(isinstance(X, np.ndarray)):
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
        self.continuous_features = [col for col in X.columns if X[col].unique().shape[0] > 2]
        self.categorical_features = [col for col in X.columns if X[col].unique().shape[0] <=2]
        #print(f"Continuous features: {self.continuous_features}")
        #print(f"Categorical features: {self.categorical_features}")
        if(self.mode == 'continuous' and len(self.continuous_features) < 3):
            print("Not enough continuous features for 'continuous' mode. Switching to 'discrete' mode.")
        #print("Mode:", self.mode)
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
        if(isinstance(X, np.ndarray)):
            X = pd.DataFrame(X)
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



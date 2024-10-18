import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class activation:
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
        # TODO
        pass
    def SGD(self, dW, db):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
    def Adam(self, dW, db):
        # TODO
        pass
    def RMSprop(self, dW, db):
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
            if(i%(self.iterations/20)==0):
                print(f'Epoch {i + 1}/{self.iterations}, Loss: {loss}')
            self.loss.append(loss)
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_hat]
    
    def predict_proba(self, X):
        z = self.linear(X)
        return self.sigmoid(z)

    
  
class MLPClassifier(Classifier):
    def __init__(self, layers=[10, 3], activate_function=activation.sigmoid, activate_derivative=activation.sigmoid_derivative, optimizer=None, learning_rate=0.05, n_epoch=10000):
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
                dz = np.dot(dz, self.W[i].T) * self.activate_derivative(self.a[i])
        
    def update(self):
        """ The update method to update parameters """
    
        if self.optimizer == 'adam':
            if not hasattr(self, 'm'):
                self.m = [np.zeros_like(w) for w in self.W]
                self.v = [np.zeros_like(w) for w in self.W]
                self.t = 0

            self.t += 1
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8

            for i in range(len(self.W)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * self.dW[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (self.dW[i] ** 2)

                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)

                self.W[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                self.b[i] -= self.learning_rate * self.db[i]
        elif self.optimizer == 'sgd':
            for i in range(len(self.W)):
                self.W[i] -= self.learning_rate * self.dW[i]
                self.b[i] -= self.learning_rate * self.db[i]
    def binaryCrossEntropy(self, y_hat, y):
        # add an epsilon to avoid taking the log of zero
        m = y.shape[0]
        epsilon = 1e-8
        loss = -1/m * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
        return loss
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        data_len = X_train.shape[0]
        # Split the data into training and validation sets
        split = int(data_len * 0.8)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

        # Initialize weights and biases
        self.W = []
        self.b = []
        self.val_loss = []
        self.layers = [X_train.shape[1]] + self.hidden_layers + [1]
        y_train = np.array(y_train).flatten()  # make sure y is one dimension
        
        # Weight initialization should be based on the layer size
        # He initialization
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i]))  # He initialization
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

            if(epoch%(self.n_epoch/5)==0):
                self.learning_rate*=0.9
            if(epoch%(self.n_epoch/10)==0):
                
                print(f'Epoch {epoch + 1}/{self.n_epoch}, Loss: {loss}, Val Loss: {val_loss if X_val is not None else "N/A"}')
            
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



    
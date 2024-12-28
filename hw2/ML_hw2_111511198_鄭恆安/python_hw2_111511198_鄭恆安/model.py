import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
# ====== Activation funtion ====== #
class activation:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        """ The sigmoid function """
        print(type(x))
        x = np.array(x)  # Ensure x is a numpy array
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



    
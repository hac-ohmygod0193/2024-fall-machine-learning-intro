import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier, LogisticRegressionClassifier, activation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os


def dataPreprocessing(threshold=0.1):
    """ TODO, use your own dataPreprocess function here. """
     
    root_path = './proj2_data/' # change the root path 
    
    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")
    test_y = os.path.join(root_path, "test_y.csv")

    train_X = pd.read_csv(train_X)
    train_y = pd.read_csv(train_y)
    test_X = pd.read_csv(test_X)
    test_y = pd.read_csv(test_y)

    # preprocess the data
    # standardize & booleanize the data

    train_X = Preprocessor(train_X).preprocess()
    test_X = Preprocessor(test_X).preprocess()

    # use feature_selection to select the feature
    
    # concat train_X and train_y
    train_data = pd.concat([train_X, train_y], axis=1)
    train_data = train_data.fillna(train_data.mean()) # 0 here stands for np.mean() in original data
    selected_column = Preprocessor(train_data).feature_selection(threshold)
    train_data = train_data[selected_column]

    # balance the data
    train_data = Preprocessor(train_data).oversample_minority_class()
    
    # concat test_X and test_y
    test_data = pd.concat([test_X, test_y], axis=1)
    test_data = test_data.fillna(test_data.mean())
    test_data = test_data[selected_column]

    # print(train_data.head())

    # convert the data to numpy array
    train_X = train_data.iloc[:, :-1].to_numpy().astype(float)
    train_y = train_data.iloc[:, -1].to_numpy().astype(float)
    
    test_X = test_data.iloc[:, :-1].to_numpy().astype(float)
    test_y = test_data.iloc[:, -1].to_numpy().astype(float)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    
    train_X, train_y, test_X, test_y = dataPreprocessing(0.1) # train, test data should not contain index

    #model = LogisticRegressionClassifier() # remember to change the hyperparameter
    model = MLPClassifier(layers= [20,10,5], activate_function=activation.sigmoid, activate_derivative=activation.sigmoid_derivative, optimizer='adam', learning_rate=0.05, n_epoch = 100000)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print(pred)
    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    


import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from model import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score

def dataPreprocessing():
    """
    Preprocesses the training and testing data for a machine learning project.
    This function performs the following steps:
    1. Reads training and testing data from CSV files.
    2. Applies preprocessing to the training and testing features.
    3. Concatenates the features and labels for both training and testing data.
    4. Fills any missing values with 0.
    5. Selects features based on a specified threshold.
    6. Converts the processed data into numpy arrays.
    Returns:
        tuple: A tuple containing four numpy arrays:
            - train_X (numpy.ndarray): Preprocessed training features.
            - train_y (numpy.ndarray): Training labels.
            - test_X (numpy.ndarray): Preprocessed testing features.
            - test_y (numpy.ndarray): Testing labels.
    """

    """ TODO, implement your own dataPreprocess function here. """
    train_X = pd.read_csv('proj1_data/train_X.csv') 
    train_y = pd.read_csv('proj1_data/train_y.csv')
    test_X = pd.read_csv('proj1_data/test_X.csv')
    test_y = pd.read_csv('proj1_data/test_y.csv')
    
    # preprocess the data
    # standardize & booleanize the data
    train_X = Preprocessor(train_X).preprocess()
    test_X = Preprocessor(test_X).preprocess()

    # use feature_selection to select the feature
    
    # concat train_X and train_y
    train_data = pd.concat([train_X, train_y], axis=1)
    train_data = train_data.fillna(0) # 0 here stands for np.mean() in original data
    selected_column = Preprocessor(train_data).feature_selection(0.11)
    train_data = train_data[selected_column]

    # concat test_X and test_y
    test_data = pd.concat([test_X, test_y], axis=1)
    test_data = test_data.fillna(0)
    test_data = test_data[selected_column]

    print(train_data.head())

    # convert the data to numpy array
    train_X = train_data.iloc[:, :-1].to_numpy().astype(float)
    train_y = train_data.iloc[:, -1].to_numpy().astype(float)
    
    test_X = test_data.iloc[:, :-1].to_numpy().astype(float)
    test_y = test_data.iloc[:, -1].to_numpy().astype(float)
    
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index
    model = LogisticRegressionClassifier()
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    print(f'Pred: {pred}')
    prob = model.predict_proba(test_X)
    prob = [f'{x:.5f}' for x in prob]
    # print(f'Prob: {prob}')
    print(f'Acc: {accuracy_score(pred, test_y):.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    


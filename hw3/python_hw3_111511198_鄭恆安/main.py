import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
import os


def dataPreprocessing(threshold=0.1):
    """ TODO, use your own dataPreprocess function here. """
     
    root_path = './proj3_data/' # change the root path 
    
    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")
    
    train_X = pd.read_csv(train_X)
    train_y = pd.read_csv(train_y)
    test_X = pd.read_csv(test_X)
    

    # preprocess the data
    # standardize & booleanize the data

    train_X = Preprocessor(train_X).preprocess()
    test_X = Preprocessor(test_X).preprocess()
    print(train_X.columns)
    print(train_y.columns)
    print(test_X.columns)
    # use feature_selection to select the feature
    
    # concat train_X and train_y
    train_data = pd.concat([train_X, train_y], axis=1)
    train_data = train_data.fillna(train_data.mean()) # 0 here stands for np.mean() in original data
    selected_column = Preprocessor(train_data).feature_selection(threshold)
    train_data = train_data[selected_column]

    # balance the data
    train_data = Preprocessor(train_data).oversample_minority_class()
    
    # test data
    test_X = test_X.fillna(test_X.mean())

    # remove the label column, because we don't have test_y
    test_X = test_X[selected_column[:-1]] 

    # print(train_data.head())
    if 'Unnamed: 0' in train_X.columns:
        train_X = train_X.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in train_y.columns:
        train_y = train_y.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_X.columns:
        test_X = test_X.drop(columns=['Unnamed: 0'])
    
    
    return train_X, train_y, test_X # train, test data should be numpy array




def main():
    train_X, train_y, test_X = dataPreprocessing(0.1) # train, test data should not contain index
    print(train_X.columns)
    print(train_y.columns)
    print(test_X.columns)
    Decision_tree = DecisionTreeClassifier(max_depth=1)
    Decision_tree.fit(train_X,train_y)
    Decision_tree.print_tree()
    '''
    # TODO 
    # build your decision tree
    # predict the output of the testing data
    # remember to save the predict label as .csv file
    pred = Decision_tree.predict(test_X)
    pred = pd.DataFrame(pred)
    root_path = './' # change the root path
    pred.to_csv(root_path+'pred_hw3_111511198_鄭恆安.csv', index=False, header=False)
    '''

if __name__ == "__main__":
    np.random.seed(0)
    main()
    


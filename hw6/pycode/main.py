import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model.meta_learner import StackingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import os


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
     
    return



def main():
    root_path = r"yourtraining_data" # change the root path 
    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")
    
    fold = KFold(n_splits=10, random_state=42, shuffle=True)
    # TODO
    # implement K-Fold CV 

    model = StackingClassifier()
    model.fit(train_X, train_y)

    # TODO 
    # build your Stacking model
    # predict the output of the testing data
    # remember to paste the result of K-fold CV to your report
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    


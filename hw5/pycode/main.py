import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
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

    model = NaiveBayesClassifier()
    model.fit(train_X,train_y)

    # TODO 
    # build your NB model
    # predict the output of the testing data
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    


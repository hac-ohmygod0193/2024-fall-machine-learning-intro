# preprocessor.py
import pandas as pd
import numpy as np

class Preprocessor:
    """
    A class used to preprocess a DataFrame for machine learning tasks.

    Attributes
    ----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.

    Methods
    -------
    __init__(df)
        Initializes the Preprocessor with a DataFrame.
    
    standardize()
        Standardizes the DataFrame by scaling features F1 to F17.
    
    booleanize()
        Converts categorical features F18 to F77 to boolean values.
    
    drop_column(column)
        Drops a specified column from the DataFrame.
    
    feature_selection(threshold)
        Selects features based on their correlation with the 'label' column.
    
    preprocess()
        Applies standardization and booleanization to the DataFrame.
    """
    """ TODO """ 
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df
    def standardize(self):
        # Standardize the DataFrame
        # only F1 to F17 will be Standardize
        self.df.iloc[:, 1:18] = (self.df.iloc[:, 1:18] - self.df.iloc[:, 1:18].mean()) / self.df.iloc[:, 1:18].std()
        return self.df
    def booleanize(self):
        # Booleanize the DataFrame
        # F18 to F77 will be Booleanize
        for i in range(18, 78):
            self.df.iloc[:, i] = self.df.iloc[:, i].apply(lambda x: 1 if x == 'Yes' else 0)
        return self.df
    def drop_column(self, column):
        # Drop the column from the DataFrame
        self.df = self.df.drop(columns=[column])
        return self.df
    def feature_selection(self, threshold):
        # Return the correlation between features
        # find absolute correlation >= threshold
        column = self.df.corr()['label'][abs(self.df.corr()['label']) >= threshold].index
        # get the column name
        column = column.tolist()   
        return column   
    def preprocess(self):
        # Preprocess the DataFrame
        self.standardize()
        self.booleanize()
        return self.df
    
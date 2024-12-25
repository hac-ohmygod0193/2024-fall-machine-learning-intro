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
        # only F1 to F77 will be Standardize
        self.df.iloc[:, 1:18] = (self.df.iloc[:, 1:18] - self.df.iloc[:, 1:18].mean()) / self.df.iloc[:, 1:18].std()
        return self.df
    def normalize(self):
        # Normalize the DataFrame
        # only F1 to F77 will be Normalized
        self.df.iloc[:, 1:18] = (self.df.iloc[:, 1:18] - self.df.iloc[:, 1:18].min()) / (self.df.iloc[:, 1:18].max()-self.df.iloc[1:18].min())
        return self.df
    def booleanize(self):
        # Booleanize the DataFrame

        # Boolean features (F18-F77): 
        # F18 to F67, F60 to F72 are categorical features, each with a value ‘Yes’ or ‘No’, 
        # F68 with a value ‘Male’ or ‘Female’, 
        # F73 to F77 with a value ‘Infected’ or ‘Non-infected’. 
        for i in range(18, 68):
            self.df.iloc[:, i] = self.df.iloc[:, i].apply(lambda x: 1 if x == 'Yes' else 0)
        self.df.iloc[:, 68] = self.df.iloc[:, 68].apply(lambda x: 1 if x == 'Male' else 0)
        for i in range(69, 73):
            self.df.iloc[:, i] = self.df.iloc[:, i].apply(lambda x: 1 if x == 'Yes' else 0)
        for i in range(73, 78):
            self.df.iloc[:, i] = self.df.iloc[:, i].apply(lambda x: 1 if x == 'Infected' else 0)
        
        return self.df
    def shuffle(self):
        # Shuffle the DataFrame
        self.df = self.df.sample(frac=1).reset_index(drop=True)
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
    def oversample_minority_class(self):
        """ Oversample the minority class to balance the dataset. """
        # Separate the classes
        majority_class = self.df[self.df['label'] == 0]
        minority_class = self.df[self.df['label'] == 1]
        
        # Oversample the minority class
        minority_upsampled = minority_class.sample(len(majority_class), replace=True, random_state=42)
        
        # Combine majority class with upsampled minority class
        balanced_data = pd.concat([majority_class, minority_upsampled])
        
        # Shuffle the balanced dataset
        self.df = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return self.df   
    def preprocess(self):
        # Preprocess the DataFrame
        self.booleanize()
        #self.standardize()
        return self.df
    
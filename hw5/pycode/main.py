import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import os
pd.set_option('future.no_silent_downcasting', True)

def dataPreprocessing(threshold=0.1):
    """ TODO, use your own dataPreprocess function here. """
     
    root_path = './proj5_data/' # change the root path 
    
    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")
    
    train_X = pd.read_csv(train_X)
    train_y = pd.read_csv(train_y)
    test_X = pd.read_csv(test_X)

    # preprocess the data
    train_X = Preprocessor(train_X).preprocess()
    test_X = Preprocessor(test_X).preprocess()

    
    # use feature_selection to select the feature
    
    # concat train_X and train_y
    train_data = pd.concat([train_X, train_y], axis=1)
    train_data = train_data.fillna(train_data.mean()) # 0 here stands for np.mean() in original data
    selected_column = Preprocessor(train_data).feature_selection(threshold)

    train_data = train_data[selected_column]
    # balance the data
    #train_data = Preprocessor(train_data).oversample_minority_class()
    
    # train data
    train_X = train_data.drop(columns=['label'])
    train_y = pd.DataFrame(train_data['label'])
    # test data
    test_X = test_X[selected_column[:-1]].fillna(test_X.mean())

    # remove the label column, because we don't have test_y
    #test_X = test_X[selected_column[:-1]] 

    # print(train_data.head())
    
    if 'Unnamed: 0' in train_X.columns:
        train_X = train_X.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_X.columns:
        test_X = test_X.drop(columns=['Unnamed: 0'])
        
    
    
    return train_X, train_y, test_X # train, test data should be numpy array
def KFold_cross_validation(X, y, n, k=5):
    fold_size = len(X) // n
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    scores = {
        'accuracy': [],
        'f1': [],
        'mcc': [],
    }
    print('k:', k)
    for i in tqdm(range(n)):
        start, end = i * fold_size, (i + 1) * fold_size
        val_data = data[start:end]
        train_data = pd.concat([data[:start], data[end:]])
        
        X_train, y_train = train_data.iloc[:,:-1], pd.DataFrame(train_data.iloc[:,-1])
        X_val, y_val = val_data.iloc[:,:-1], val_data.iloc[:,-1]
        
        
        model = NaiveBayesClassifier(mode='continuous')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores['accuracy'].append(accuracy_score(y_val, y_pred))
        scores['f1'].append(f1_score(y_val, y_pred))
        scores['mcc'].append(matthews_corrcoef(y_val, y_pred))
        
        #scores['score'] = model.predict_score(y_pred, y_val)
    
    for metric in scores:
        scores[metric] = np.mean(scores[metric])
    scoring = 0.3 * scores['accuracy'] + 0.35 * scores['f1'] + 0.35 * scores['mcc']
    scores['scoring'] = scoring
    print('Kfold scoring:', scoring)
    print('\n')
    return scores






def main():
    X, y, test_X = dataPreprocessing(0.1) # train, test data should not contain index

    # split the training data into training and validation data
    total = len(X)
    train_size = int(total * 0.8)
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    KFold_cross_validation(train_X, train_y, 5)
    model = NaiveBayesClassifier(mode='continuous') # 'continuous', 'discrete' or 'mixed'
    model.fit(train_X, train_y)
    model.plot_continuous_distribution(X)
    model.plot_categorical_distribution(X)
    # TODO 
    # build your NB model
    # predict the output of the testing data
    # remember to save the predict label as .csv file
    pred = model.predict(val_X)
    print(pred)
    scoring = model.predict_score(pred, val_y.values)
    print(f'Scoring: {scoring:.5f}\n')
    probas = model.predict_proba(test_X)
    print("Class probabilities:", probas[1])

    # predict the output of the testing data
    # remember to save the predict label as .csv file
    test_pred = model.predict(test_X)
    df = pd.DataFrame()
    df['label'] = test_pred
    root_path = './' # change the root path
    df.to_csv(root_path+'pred_hw5_111511198_鄭恆安.csv')
    print('Predict label has been saved as pred_hw5_111511198_鄭恆安.csv')

    
if __name__ == "__main__":
    np.random.seed(0)
    main()
    


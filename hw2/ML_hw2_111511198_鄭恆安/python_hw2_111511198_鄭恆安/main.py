import numpy as np
import pandas as pd
from preprocessor import Preprocessor
#from model import MLPClassifier, activation, optimizer
from base_learner import MLPClassifier, activation, optimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def dataPreprocessing(threshold=0.11, root_path="./proj2_data"):
    
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

def KFold_cross_validation(X, y, n, model):
    fold_size = len(X) // n
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    scores = {
        'accuracy': [],
        'f1': [],
        'mcc': [],
    }
    print('k:', n)
    for i in tqdm(range(n)):
        start, end = i * fold_size, (i + 1) * fold_size
        val_data = data[start:end]
        train_data = pd.concat([data[:start], data[end:]])
        
        X_train, y_train = train_data.iloc[:,:-1], pd.DataFrame(train_data.iloc[:,-1])
        X_val, y_val = val_data.iloc[:,:-1], val_data.iloc[:,-1]
        
        
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
    
    train_X, train_y, test_X = dataPreprocessing(0.1) # train, test data should not contain index
    model = MLPClassifier(layers= [20,10], activate_function=activation.sigmoid, activate_derivative=activation.sigmoid_derivative, optimizer=optimizer.Adam, learning_rate=0.005, n_epoch = 10000)
    #KFold_cross_validation(train_X, train_y, 5, model)
    #'''
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    #print(pred)

    test_y = pd.read_csv('./proj2_data/test_y.csv')
    test_y = test_y['label'].values
    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')
    #'''

if __name__ == "__main__":
    np.random.seed(0)
    main()
    


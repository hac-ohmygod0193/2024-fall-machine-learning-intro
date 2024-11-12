import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


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
def KFold_cross_validation(X, y, k, max_depth=25, min_samples_leaf=3, method='entropy'):
    fold_size = len(X) // k
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    scores = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'mcc': [],
        'roc_auc': []
    }
    
    for i in tqdm(range(k)):
        start, end = i * fold_size, (i + 1) * fold_size
        val_data = data[start:end]
        train_data = pd.concat([data[:start], data[end:]])
        
        X_train, y_train = train_data.iloc[:,:-1], pd.DataFrame(train_data.iloc[:,-1])
        X_val, y_val = val_data.iloc[:,:-1], val_data.iloc[:,-1]
        
        model = DecisionTreeClassifier(max_depth, min_samples_leaf, method)
        model.fit(X_train, y_train)
        model.post_prune(X_val, y_val)
        y_pred = model.predict(X_val)
        
        scores['accuracy'].append(accuracy_score(y_val, y_pred))
        scores['f1'].append(f1_score(y_val, y_pred))
        scores['precision'].append(precision_score(y_val, y_pred))
        scores['recall'].append(recall_score(y_val, y_pred))
        scores['mcc'].append(matthews_corrcoef(y_val, y_pred))
        scores['roc_auc'].append(roc_auc_score(y_val, y_pred))
        #scores['score'] = model.predict_score(y_pred, y_val)
    
    for metric in scores:
        scores[metric] = np.mean(scores[metric])
    scoring = 0.3 * scores['accuracy'] + 0.35 * scores['f1'] + 0.35 * scores['mcc']
    scores['scoring'] = scoring
    print('scoring:', scoring)
    print('\n')
    return scores
def find_best_params(X, y, params):
    best_params = {
        'max_depth': None,
        'min_samples_leaf': None,
        'method': None,
    }
    best_score = 0
    for max_depth in params['max_depth']:
        for min_samples_leaf in params['min_samples_leaf']:
            for method in params['method']:
                print('max_depth:', max_depth, 'min_samples_leaf:', min_samples_leaf, 'method:', method)
                score = KFold_cross_validation(X, y, 10, max_depth, min_samples_leaf, method)
                if score['scoring'] > best_score:
                    best_score = score['scoring']
                    best_params = {
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'method': method
                    }
    print('best_params:', best_params, 'best_score:', best_score)
    return best_params, best_score
def main():
    train_X, train_y, test_X = dataPreprocessing(0.1) # train, test data should not contain index

    # split the training data into training and validation data
    total = len(train_X)
    train_size = int(total * 0.8)
    train_X, val_X = train_X[:train_size], train_X[train_size:]
    train_y, val_y = train_y[:train_size], train_y[train_size:]

    params = {
        'max_depth': [15, 20, 25],
        'min_samples_leaf': [1, 3, 5],
        'method': ['entropy', 'gini']
    }
    # use KFold_cross_validation to find the best hyperparameters
    #best_params, best_score = find_best_params(train_X, train_y, params)
    #model = DecisionTreeClassifier(max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'], method=best_params['method'])
    
    # best model
    model = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5, method='gini')
    model.fit(train_X,train_y)
    
    
    
    pred = model.predict(val_X)
    scoring = model.predict_score(pred, val_y)
    print(f'Scoring: {scoring:.5f}')
    print('-' * 50)

    # post prune the tree with validation data
    print('After pruning\n\n')
    model.post_prune(val_X, val_y)

    pred = model.predict(val_X)
    scoring = model.predict_score(pred, val_y)
    print(f'Scoring: {scoring:.5f}')
    print('-' * 50)
    model.print_tree()
    print('-' * 50)
    
   
    # predict the output of the testing data
    # remember to save the predict label as .csv file
    test_pred = model.predict(test_X)
    df = pd.DataFrame()
    df['label'] = test_pred
    root_path = './' # change the root path
    df.to_csv(root_path+'pred_hw3_111511198_鄭恆安.csv')
    print('Predict label has been saved as pred_hw3_111511198_鄭恆安.csv')
    
if __name__ == "__main__":
    np.random.seed(0)
    main()
    


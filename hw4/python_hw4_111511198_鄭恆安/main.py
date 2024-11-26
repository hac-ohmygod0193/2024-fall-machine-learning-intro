import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import KNearestNeighborClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)

def dataPreprocessing(threshold=0.1):
    """ TODO, use your own dataPreprocess function here. """
     
    root_path = './proj4_data/' # change the root path 
    
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
    train_data = Preprocessor(train_data).oversample_minority_class()
    
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

def KFold_cross_validation(X, y, n, k=3, method='euclidean'):
    fold_size = len(X) // n
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    scores = {
        'accuracy': [],
        'f1': [],
        'mcc': [],
    }
    print('k:', k, 'method:', method)
    for i in tqdm(range(n)):
        start, end = i * fold_size, (i + 1) * fold_size
        val_data = data[start:end]
        train_data = pd.concat([data[:start], data[end:]])
        
        X_train, y_train = train_data.iloc[:,:-1], pd.DataFrame(train_data.iloc[:,-1])
        X_val, y_val = val_data.iloc[:,:-1], val_data.iloc[:,-1]
        
        X_train = X_train.values
        y_train = y_train.values
        X_val = X_val.values
        y_val = y_val.values

        model = KNearestNeighborClassifier(k, method)
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
    print('scoring:', scoring)
    print('\n')
    return scores
def find_best_params(X, y, params):
    best_params = {
        'k': 0,
        'method': ''
    }
    best_score = 0
    for method in params['method']:
        k_values = []
        scores_list = []
        for k in params['k']:
            scores = KFold_cross_validation(X, y, 5, k, method)
            if scores['scoring'] > best_score:
                best_score = scores['scoring']
                best_params['k'] = k
                best_params['method'] = method
            k_values.append(k)
            scores_list.append(scores['scoring'])
    print('best_params:', best_params, 'best_score:', best_score)
    return best_params, best_score
def draw_relationship(train_X, train_y,val_X, val_y, k_values): 
    if(isinstance(train_X, pd.DataFrame)):
        train_X = train_X.values
        train_y = train_y.values
        val_X = val_X.values
        val_y = val_y.values
    methods = ['manhattan', 'euclidean', 'chebyshev']
    for method in methods:
        val_scores = []
        for k in tqdm(k_values):
            model = KNearestNeighborClassifier(k=k, distance_metric=method)
            model.fit(train_X, train_y)
            val_pred = model.predict(val_X)
            val_score = accuracy_score(val_y, val_pred)
            val_error_rate = 1 - val_score
            val_scores.append(val_error_rate)
        plt.plot(k_values, val_scores, label=f'Validation Accuracy ({method})', marker='o') 
    plt.xlabel('k')
    plt.ylabel('Error Rate')
    plt.title('Validation Error Rate vs K for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    X, y, test_X = dataPreprocessing(0.11) # train, test data should not contain index

    # split the training data into training and validation data
    total = len(X)
    train_size = int(total * 0.8)
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    params = {
        'k': [i for i in range(30, 36)],
        'method': ['manhattan', 'euclidean', 'chebyshev']
    }
    """find the best k and method"""
    #k_values = range(1,40)
    #draw_relationship(train_X, train_y, val_X, val_y, k_values)

    """use KFold_cross_validation to find the best hyperparameters"""
    #best_params, best_score = find_best_params(X, y, params)
    #model = KNearestNeighborClassifier(k=best_params['k'], distance_metric=best_params['method'])
    
    model = KNearestNeighborClassifier(k=31, distance_metric='manhattan')
    model.fit(train_X.values, train_y.values)
    pred = model.predict(val_X.values)
    #print(pred)
    scoring = model.predict_score(pred, val_y.values)
    print(f'Scoring: {scoring:.5f}\n')
    
  
    
    # predict the output of the testing data
    # remember to save the predict label as .csv file
    test_pred = model.predict(test_X)
    df = pd.DataFrame()
    df['label'] = test_pred
    root_path = './' # change the root path
    df.to_csv(root_path+'pred_hw4_111511198_鄭恆安.csv')
    print('Predict label has been saved as pred_hw4_111511198_鄭恆安.csv')

if __name__ == "__main__":
    np.random.seed(0)
    main()
    


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_gamma_deviance, mean_poisson_deviance, max_error
from sklearn.metrics import pairwise, pairwise_distances
from sklearn.metrics import get_scorer
from tslearn.metrics import dtw, soft_dtw, lcss
from scipy.spatial.distance import chebyshev


def get_ds(predictions, true, size, K):
    #print(predictions)
    errors = {}
    selection = {}
    selection_preds = {}
    for w in range(size,len(predictions)):
        first = w - size    
        last = w
        errors[w] = {}        
        #selection[w] = {}        
        selection[w] = []
        for col in predictions.columns:
           #print(col)
           predictions_w = predictions.iloc[first:last][col]
           true_w = true.iloc[first:last]
           #print(true_w)
           #print(predictions_w)
           errors[w][col] = mean_squared_error(true_w, predictions_w)
        rank = pd.Series(errors[w]).rank()
        selection_array = np.zeros(K)
        #print(rank)
        for i in range(K):
            try:
                #selection[w][i] = rank.loc[rank == i+1].index.values[0]
                selection[w].append(int(rank.loc[rank == i+1].index.values[0]))
            except:
                #print(['*']*1000)
                #print(df_error.idxmin()) # solucao para ranks malucos 1.5, 2 sem 1...
                selection[w].append(rank.idxmin())
            selection_array[i] = predictions.iloc[w][selection[w][i]]
            selection_preds[w] = selection_array.mean()
    return pd.Series(selection).reset_index(drop=True), pd.Series(selection_preds).reset_index(drop=True)


def get_ds2(predictions, true, metric, size, K):
    #print(predictions)
    errors = {}
    selection = {}
    selection_preds = {}
    for w in range(size,len(predictions)):
        first = w - size    
        last = w
        errors[w] = {}        
        #selection[w] = {}        
        selection[w] = []
        for col in predictions.columns:
           #print(col)
           predictions_w = predictions.iloc[first:last][col]
           true_w = true.iloc[first:last]
           #print(true_w)
           #print(predictions_w)
           #errors[w][col] = metric(true_w, predictions_w)
           errors[w][col] = get_mtr(metric, true_w, predictions_w)
        rank = pd.Series(errors[w]).rank()
        selection_array = np.zeros(K)
        #print(rank)
        for i in range(K):
            try:
                #selection[w][i] = rank.loc[rank == i+1].index.values[0]
                selection[w].append(int(rank.loc[rank == i+1].index.values[0]))
            except:
                #print(['*']*1000)
                #print(df_error.idxmin()) # solucao para ranks malucos 1.5, 2 sem 1...
                selection[w].append(rank.idxmin())
            selection_array[i] = predictions.iloc[w][selection[w][i]]
            selection_preds[w] = selection_array.mean()
    return pd.Series(selection).reset_index(drop=True), pd.Series(selection_preds).reset_index(drop=True)

def get_mtr(name:str, y_true, y_pred):
    if (name == 'mse'):
        return mean_squared_error(y_true, y_pred)
    if (name == 'mselog'):
        return mean_squared_log_error(y_true, y_pred)
    if (name == 'rmse'):
        return mean_squared_error(y_true, y_pred, squared=False)
    if (name == 'mae'):
        return mean_absolute_error(y_true, y_pred)
    if (name == 'max'):
        return max_error(y_true, y_pred)
    if (name == 'gamma'):
        return mean_gamma_deviance(y_true, y_pred)
    if (name == 'poisson'):
        return mean_poisson_deviance(y_true, y_pred)
    if (name == 'r2'):
        return r2_score(y_true, y_pred)
    if (name == 'dtw'):
        return dtw(y_true, y_pred)
    if (name == 'cosine'):
        return pairwise.paired_cosine_distances(y_true.values.reshape(1,-1), y_pred.values.reshape(1,-1))
    if (name == 'euclidean'):
        return pairwise.paired_euclidean_distances(y_true.values.reshape(1,-1), y_pred.values.reshape(1,-1))
    if (name == 'manhattan'):
        return pairwise.paired_manhattan_distances(y_true.values.reshape(1,-1), y_pred.values.reshape(1,-1))
    if (name == 'chebyshev'):
        return chebyshev(y_true.values.reshape(1,-1), y_pred.values.reshape(1,-1))
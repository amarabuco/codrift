import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
import sklearn
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import torch

import mlflow
import mlflow.statsmodels
from mlflow import log_metric, log_param, log_metrics, log_params, log_artifact, log_artifacts
import optuna
from urllib.parse import urlparse


def get_data(folder, filename):
    if folder == 'covid': 
       df = pd.read_csv('./data/'+folder+'/'+filename, index_col=0)
       #df = df.loc[df.index < '2022-01-01']
       df = df.iloc[:710]
    return df


def fix_outliers(df,q = .99, zero = True):
  # cortar período inicial sem casos
  for k,d in enumerate(df):
    if k > 0 and d > df.iloc[k-1]:
      break
  df = df.iloc[k:]
  # substituir zeros
  if zero == False:
    df = df.mask(df == 0, np.nan).fillna(method='bfill')
  # converter valores extremos para NaN e substituir pelo anterior
  df = df.mask(df > df.quantile(q), np.nan).fillna(method='bfill')
  df = df.mask(df < 0, np.nan).fillna(0)
  return df

def train_val_test_split(data, train_size:float, val_size:float):
    train_data = data.iloc[:int(len(data)*train_size)] 
    val_data = data.iloc[int(len(data)*train_size):int(len(data)*(train_size+val_size))]
    test_data = data.iloc[-int(len(data)*(1-train_size-val_size)):]
    return train_data, val_data, test_data

def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    #mape = mean_absolute_percentage_error(actual, pred)
    #print(actual)
    #print(pred)
    #print(actual.loc[actual[actual.columns[0]] > 0])
    actual_adj = actual.loc[actual[actual.columns[0]] > 0]
    pred_adj = pred.loc[actual_adj.index]
    mape = mean_absolute_percentage_error(actual_adj, pred_adj)
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2':r2}

def splitter(data, lags):
    X = pd.DataFrame(tsa.add_lag(data, lags=lags))
    y = X.pop(0)
    return X, y

def normalizer(input, fit=True, transform=1):
    """ pass normalizer to fit """
    MinMax = preprocessing.MinMaxScaler(feature_range=(-1,1))
    if fit == True:
        out = MinMax.fit_transform(input)
        return MinMax, out
    else:
        if transform == -1:
            out = fit.inverse_transform(input)
        else:
            out = fit.transform(input)
        return out

def torcher(data):
    #print(data)
    #print(torch.from_numpy(data).float())
    return torch.from_numpy(data).float()


def torch_data(train_data, val_data, test_data, lags):

    #train_data, val_data, test_data = train_val_test_split(get_data(), 0.7, 0.2)

    X_train = pd.DataFrame(tsa.add_lag(train_data, lags =lags))
    #X_train = pd.DataFrame(tsa.add_lag(train_data.diff().dropna(), lags =14))
    y_train = torch.from_numpy(X_train.pop(0).values.reshape(-1,1)).float()

    X_val = pd.DataFrame(tsa.add_lag(val_data, lags =lags))
    #X_train = pd.DataFrame(tsa.add_lag(train_data.diff().dropna(), lags =14))
    y_val = torch.from_numpy(X_val.pop(0).values.reshape(-1,1)).float()

    #X_test = pd.DataFrame(tsa.add_lag(test_data, lags =14))
    X_test = pd.DataFrame(tsa.add_lag(test_data, lags =lags))
    y_test = torch.from_numpy(X_test.pop(0).values.reshape(-1,1)).float()

    #normalization ELM
    min_value = float(train_data.min())
    base = float(train_data.max()) - float(train_data.min())

    X_train_mm = ((X_train - min_value)/base).to_numpy()
    X_train_mm = torch.from_numpy(X_train_mm).float()
    X_val_mm = ((X_val - min_value)/base).to_numpy()
    X_val_mm = torch.from_numpy(X_val_mm).float()
    X_test_mm = ((X_test - min_value)/base).to_numpy()
    X_test_mm = torch.from_numpy(X_test_mm).float()

    return X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test

def post_forecast(preds):
    return preds.mask(preds < 0, 0)

def log_arts(country,model):
    log_artifacts("outputs/"+country+"/data")
    log_artifacts("outputs/"+country+"/preds/"+model)
    log_artifact('outputs/'+country+'/prediction.png')
    return True

def create_dirs(country, model):
    try:
        if not os.path.exists("outputs/"+country+"/data"):
            os.makedirs("outputs/"+country+"/data")
        if not os.path.exists("outputs/"+country+"/preds/"+model):
            os.makedirs("outputs/"+country+"/preds/"+model)
        if not os.path.exists("models/"+country):
            os.makedirs("models/"+country)
    except:
        pass
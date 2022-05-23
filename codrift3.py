#Dependencies
""" 
!pip install -U sklearn
!pip install pmdarima
!pip install river
!pip install tslearn
!pip install arch
!pip install skorch
"""

#Imports

import json
import math
import calendar
from datetime import timedelta 
from datetime import datetime as dt
import numpy as np
import pandas as pd
import warnings
import os
import sys
warnings.simplefilter('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import mlflow
import mlflow.statsmodels
from mlflow import log_metric, log_param, log_metrics, log_params, log_artifact, log_artifacts
import optuna
from urllib.parse import urlparse

from pmdarima.arima import auto_arima as pmautoarima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pickle
import joblib
import sklearn
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers

from river import drift

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def draw_data(country, train,val,test):
    fig, ax = plt.subplots()
    ax.plot(train, color='black')
    ax.plot(val, color='darkgray')
    ax.plot(test, color='lightgray')
    plt.savefig('outputs/'+country+'/data.png')
    # Log an artifact (output file)   
    return True

def get_drifts(data, col, detector='adwin'):
    if (detector == 'adwin'):
        drift_detector = drift.ADWIN(delta=0.001)
    
    data_diff = data.to_dict()[col]
    drifts =[]
    drift_data = {}
   
    for k in data_diff:
        #print(k)
        in_drift, in_warning = drift_detector.update(data_diff[k])
        if in_drift:
            drifts.append(k) 
            #print(f"Change detected at index {k}, input value: {data[k]}")  
    for key, drift_point in enumerate(drifts):
        if key == 0:
            drift_data[key] = data[:drift_point]
        elif key == len(drifts)-1:
            drift_data[key] = data[drifts[key-1]:drift_point]
            drift_data[key+1] = data[drift_point:]
        else:
            drift_data[key] = data[drifts[key-1]:drift_point]
    
    return drifts, drift_data
    
def draw_drifts(country, drifts, drift_data, train_data):
  fig, (ax1, ax2) = plt.subplots(2,1,figsize=(18,8), sharex=True)
  for d in range(len(drift_data)):
    ax1.plot(drift_data[d].fillna(method='bfill'))
  
  #print(drifts)
  #print(train_data.loc[drifts].values[0])
  #print(train_data.loc[drifts].values[0,:])
  #print(train_data.loc[drifts].values[:,0])
  ax1.bar(x=train_data.loc[drifts].index, height=train_data.loc[drifts].values[:,0], width=2, color='r')
  #ax1.annotate(train_data.loc[drifts].index, xy=(10, -100),  xycoords='axes points', xytext=(train_data.loc[drifts].index, -150), textcoords='data')
  """
  for drift_point in pd.to_datetime(drifts, format="%m/%d/%y"):
      print(drift_point)
      print(drift_point.date())
      #ax1.annotate(k, xy=(10, 100),  xycoords='axes points', xytext=(drift_point, -10), textcoords='data')
      #ax1.annotate(drift_point.date().strftime('%Y-%m-%d'), xy=(10, -100),  xycoords='axes points', xytext=(drift_point-delta(days=10), -150), textcoords='data', rotation=90)
      ax1.annotate(drift_point.date(), xy=(10, -100),  xycoords='axes points', xytext=(train_data.loc[drifts].index, -150), textcoords='data')
      #ax1.annotate(k, xy=(10, -5000),  xycoords='axes points', xytext=(dt.strptime(drift_point, "%m/%d/%y")-timedelta(days=20), -5500), textcoords='data')
  """
  ax2.plot(train_data.cumsum())
  plt.savefig('outputs/'+country+'/drifts.png')
  # Log an artifact (output file)

  return True


def random_walk(data):
    return data.shift(1).dropna()

def train_arima(train_data, sarima=False):
    if sarima == True:
        arima_order = pmautoarima(train_data, max_p=7, d=1, max_q=7, m=7, seasonal=True, trace=True, information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
        arima_base = ARIMA(train_data, order=arima_order.order, seasonal_order=arima_order.seasonal_order)
        #log_param("order", arima_order.order)
        #log_param("seasonal_order", arima_order.seasonal_order)
    else:
        arima_order = pmautoarima(train_data, max_p=7, d=1, max_q=7, m=7, seasonal=False, trace=True, information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
        arima_base = ARIMA(train_data, order=arima_order.order)
        #log_param("order", arima_order.order)
    arima_model = arima_base.fit()
    return arima_model

def update_arima(model, test_data):
    arima_updated = model.apply(test_data)
    return arima_updated

class ELM():
    def __init__(self, input_size, h_size, activation, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = 1
        self._device = device
        self.activation_name = activation

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = torch.zeros(self._h_size, device=self._device)

        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)

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

def objective_elm(trial):
    #h_size = trial.suggest_int("h_size", 2, 20)
    h_size = trial.suggest_categorical('h_size', [8, 16, 32, 64, 100, 200])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh", "relu"])
    # Generate the model.
    results = np.zeros(10)
    
    for i in range(10):
        model = ELM(input_size=14, h_size=h_size, activation=activation, device=device) #variar o input-size (FIX)
        model.fit(X_train_mm, y_train)
        y_pred = model.predict(X_val_mm)
        mse = -mean_squared_error(y_val, y_pred)
        results[i] = mse

    return results.mean()

def train_elm(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='elm_study')
    study.optimize(objective_elm, n_trials=50, show_progress_bar=True)
    params = study.best_params
    best_elm = ELM(input_size=14, h_size=params['h_size'], activation=params['activation'], device=device) #variar o input-size (FIX)
    best_elm.fit(x, y)
    return best_elm

def objective_svm(trial):
    svr_k = trial.suggest_categorical('kernel',['linear', 'rbf'])
    #svr_k = trial.suggest_categorical('kernel',['rbf'])
    svr_g = trial.suggest_categorical("gamma", [1, 0.1, 0.01, 0.001]) #auto, scale
    #svr_g = trial.suggest_float("gamma", 0.001, 1, log=True) #auto, scale
    svr_c = trial.suggest_categorical("C", [0.1, 1, 100, 1000, 10000])
    #svr_c = trial.suggest_float("C", 0.1, 10000, log=True)
    svr_e = trial.suggest_categorical("epsilon", [0.1, 0.01, 0.001])
    #svr_e = trial.suggest_float("epsilon",0.001, 0.1, log=True)
    svr_t = trial.suggest_categorical("tolerance", [0.01, 0.001, 0.0001])
    #svr_t = trial.suggest_float("tolerance", 0.0001, 0.01, log=True)
    regressor_obj = sklearn.svm.SVR(kernel=svr_k, gamma=svr_g, C=svr_c, epsilon=svr_e, tol=svr_t)

    score = sklearn.model_selection.cross_val_score(regressor_obj, np.concatenate((X_train_mm, X_val_mm)), np.concatenate((y_train, y_val)),
                                                    n_jobs=-1, cv=TimeSeriesSplit(2, test_size=len(y_val)//2),
                                                    scoring='neg_root_mean_squared_error').mean()
    return score

def train_svm(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='svr_study')
    study.optimize(objective_svm, n_trials=100, show_progress_bar=True)
    params = study.best_params
    best_svr = SVR(kernel=params['kernel'], C=params['C'], epsilon=params['epsilon'], tol=params['tolerance'])
    best_svr.fit(x, y)
    return best_svr


def objective_lstm(trial):
    model = Sequential()
    model.add(layers.LSTM(units= trial.suggest_categorical('units', [8, 16, 32, 64, 100, 200]), input_shape=(14, 1)))
    #model.add(layers.LSTM(100, input_shape=(14, 1), dropout=0.2, return_sequences=True))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.LSTM(100, dropout=0.2))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=trial.suggest_categorical('activation', ['relu', 'linear', 'tanh']),) )

    score = np.zeros(3)
    for i in range(3):
        # We compile our model with a sampled learning rate.
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')


        model.fit(
            X_train_mm,
            y_train,
            validation_data=(X_val_mm, y_val),
            shuffle=True,
            batch_size=32,
            epochs=40,
            verbose=False,
        )

        # Evaluate the model accuracy on the validation set.
        score[i] = model.evaluate(X_val_mm, y_val, verbose=0)
    return score.mean()   

def lstm(input_size, units, activation):
    model = Sequential()
    model.add(layers.LSTM(units= units, input_shape=(input_size, 1)))
    model.add(layers.Dense(1, activation=activation ))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='lstm_study')
    study.optimize(objective_lstm, n_trials=10, show_progress_bar=True)
    params = study.best_params
    best_lstm = lstm(input_size=14, units=params['units'], activation=params['activation']) #variar o input-size (FIX)
    best_lstm.fit(x, y)
    return best_lstm


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

def draw_predictions(country, predictions, true):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(predictions.reset_index(drop=True), label='pred')
    ax.plot(true.reset_index(drop=True), label='true')
    plt.legend()
    plt.savefig('outputs/'+country+'/prediction.png')

def get_oracle(predictions, true):
  df_error = predictions.iloc[1:].rsub(np.array(true.iloc[1:]), axis=0).abs()
  oracle = {}
  selection = []
  for row in df_error.rank(axis=1).idxmin(axis=1).items():
    oracle[row[0]] = predictions.at[row[0], row[1]]
    selection.append(row[1])
  return selection, pd.Series(oracle)

def post_forecast(preds):
    return preds.mask(preds < 0, 0)

def log_arts(country,model):
    log_artifacts("outputs/"+country+"/data")
    log_artifacts("outputs/"+country+"/preds/"+model)
    log_artifact('outputs/'+country+'/prediction.png')
    return True


def main():
    
    exp = "codrift_220525"
    mlflow.set_experiment(exp)
        
    if sys.argv[1] == 'help':
        print('models: rw, arima, sarima, AS, ASDS, ASO...')
    
    country = sys.argv[1]
    model = sys.argv[2]
    split = sys.argv[3]
    lags = int(sys.argv[4])
    size = int(sys.argv[5])
    if (model == 'ASDS' or model == 'AEDS' or model == 'ASVDS'):
        K = int(sys.argv[6])
        rname = country+'.'+model+'.'+split+'.'+str(lags)+'.'+str(size)+'.'+str(K)
    else:
        rname =  country+'.'+model+'.'+split

    print(exp)
    print(rname)
    with mlflow.start_run(run_name=rname):

            if not os.path.exists("outputs/"+country+"/data"):
                os.makedirs("outputs/"+country+"/data")
            if not os.path.exists("outputs/"+country+"/preds/"+model):
                os.makedirs("outputs/"+country+"/preds/"+model)
            if not os.path.exists("models/"+country):
                os.makedirs("models/"+country)

            data = get_data('covid', country+'_daily.csv')
            data = fix_outliers(data)
            train_data, val_data, test_data = train_val_test_split(data, 0.7, 0.2)
            #print('train: ', train_data.head(1), train_data.tail(1), len(train_data) )
            #print('val: ', val_data.head(1), val_data.tail(1), len(val_data) )
            #print('test: ', test_data.head(1), test_data.tail(1), len(test_data) )

            train_data.to_csv("outputs/"+country+"/data/train_data.csv")
            val_data.to_csv("outputs/"+country+"/data/val_data.csv")
            test_data.to_csv("outputs/"+country+"/data/test_data.csv")
            draw_data(country, train_data, val_data, test_data)
            log_artifact('outputs/'+country+'/data.png')

            global X_train_mm
            global y_train
            global X_val_mm
            global y_val
            global X_test_mm
            global y_test

            X_train_mm = 0
            y_train = 0
            X_val_mm = 0
            y_val = 0
            X_test_mm = 0
            y_test = 0
            
            X, y = splitter(data, lags)
            sz = len(X)
            X_train, y_train = X[:int(sz*0.7)], y[:int(sz*0.7)]
            X_val, y_val = X[int(sz*0.7):int(sz*0.9)], y[int(sz*0.7):int(sz*0.9)]
            X_test, y_test = X[int(sz*0.9)-1:], y[int(sz*0.9)-1:]

            if not os.path.exists("models/"+country+"/normx.pkl"):
                normx, X_train_mm = normalizer(X_train)
                normy, y_train_mm = normalizer(y_train.values.reshape(-1,1))
                joblib.dump(normx, "models/"+country+"/normx.pkl") 
                joblib.dump(normy, "models/"+country+"/normy.pkl") 
            else:
                normx = joblib.load("models/"+country+"/normx.pkl")
                normy = joblib.load("models/"+country+"/normy.pkl")

            X_val_mm = normalizer(X_val, normx)
            X_test_mm = normalizer(X_test, normx)
            y_val_mm = normalizer(y_val.values.reshape(-1,1), normy)
            y_test_mm = normalizer(y_test.values.reshape(-1,1), normy)

            if model == 'rw':
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                #Baseline 
                    if split == 'val':
                        y_pred = post_forecast(random_walk(val_data))
                        metrics = eval_metrics(val_data[1:], y_pred)
                        draw_predictions(country, y_pred, val_data)
                    elif split == 'test':
                        y_pred = post_forecast(random_walk(test_data))
                        metrics = eval_metrics(test_data.iloc[size+1:], y_pred.iloc[size:])
                        draw_predictions(country, y_pred, test_data)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model})
                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    
                    log_metrics(metrics)
                    log_arts(country,model)
                    mlflow.end_run()
                    
            if model == 'arima':
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/arima.pkl"):
                        arima = train_arima(train_data)
                        arima.save("models/"+country+"/arima.pkl")
                    else:
                        arima = sm.load("models/"+country+"/arima.pkl")
                    log_params(arima.specification)
                    
                    with open("outputs/"+country+"/arima.txt", "w") as f:
                        f.write(arima.summary().as_text())
                    log_artifact("outputs/"+country+"/arima.txt")
                    if split == 'val':
                        arima = update_arima(arima, val_data)
                        y_pred = post_forecast(arima.predict())
                        metrics = eval_metrics(val_data, y_pred)
                        draw_predictions(country, y_pred, val_data)
                    elif split == 'test':
                        arima = update_arima(arima, test_data)
                        y_pred = post_forecast(arima.predict())
                        #metrics = eval_metrics(y_pred, test_data)
                        test_data.index = y_pred.index
                        print('mape', mean_absolute_percentage_error(test_data, y_pred))
                        metrics = eval_metrics(test_data.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, test_data)
                    log_metrics(metrics)
                    log_params(arima.specification)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})
                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
                    
            if model == 'sarima':
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/sarima.pkl"):
                        sarima = train_arima(train_data, sarima=True)
                        sarima.save("models/"+country+"/sarima.pkl")
                    else:
                        sarima = sm.load("models/"+country+"/sarima.pkl")
                    log_params(sarima.specification)
                    
                    if split == 'val':
                        sarima = update_arima(sarima, val_data)
                        y_pred = post_forecast(sarima.predict())
                        metrics = eval_metrics(val_data, y_pred)
                        draw_predictions(country, y_pred, val_data)
                    elif split == 'test':
                        sarima = update_arima(sarima, test_data)
                        y_pred = post_forecast(sarima.predict())
                        metrics = eval_metrics(test_data.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, test_data)
                    log_metrics(metrics)
                    log_params(sarima.specification)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    with open("outputs/"+country+"/sarima.txt", "w") as f:
                        f.write(sarima.summary().as_text())
                    log_artifact("outputs/"+country+"/sarima.txt")
                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
            
            if model == 'svm':                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    #y_train = normalizer(y_train.values.reshape(-1,1), normy).ravel()
                    X_val_mm = normalizer(X_val, normx)
                    #y_val = normalizer(y_val.values.reshape(-1,1), normy).ravel()
                    X_test_mm = normalizer(X_test, normx)

                    #print(X_train_mm)
                    #print(y_train)
                    
                    #print(np.concatenate((X_train_mm, X_val_mm), axis=0), np.concatenate((y_train, y_val), axis=0))
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/svm.pkl"):
                        svm = train_svm(X_train_mm, y_train)
                        joblib.dump(svm, "models/"+country+"/svm.pkl") 
                    else:
                        svm = joblib.load("models/"+country+"/svm.pkl")                  
                    
                    if split == 'test':
                        #y_pred = normalizer(svm.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                        y_pred = svm.predict(X_test_mm).reshape(-1,1).flatten()
                        y_pred = post_forecast(pd.DataFrame(y_pred))
                        #y_pred = post_forecast(pd.DataFrame(svm.predict(X_test_mm)))
                        y_test = pd.DataFrame(y_test.reset_index(drop=True))
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    log_params(svm.get_params())
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
                
            if model == 'elm':
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train_mm = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val_mm = normalizer(y_val.values.reshape(-1,1), normy)

                    X_test_mm = normalizer(X_test, normx)
                    X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train.values.reshape(-1,1)), torcher(X_val_mm), torcher(y_val.values.reshape(-1,1))
                    X_test_mm, y_test= torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
        
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/elm.pkl"):
                        elm = train_elm(X_train_mm, y_train)
                        torch.save(elm ,"models/"+country+"/elm.pkl")
                    else:
                        elm = torch.load("models/"+country+"/elm.pkl")
                    
                    if split == 'test':
                        y_pred = normalizer(elm.predict(X_test_mm).numpy().reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_test = pd.DataFrame(y_test.numpy()).reset_index(drop=True)
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    log_params({'h_size': elm._h_size})
                    #log_params({'h_size': elm._h_size, 'activation' :elm.activation_name})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
            
            if model == 'lstm':
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val = normalizer(y_val.values.reshape(-1,1), normy)

                    X_test_mm = normalizer(X_test, normx)
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train.values.reshape(-1,1)), torcher(X_val_mm), torcher(y_val.values.reshape(-1,1))
                    #X_test_mm, y_test= torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
        
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/lstm"):
                        lstm = train_lstm(X_train_mm, y_train)
                        lstm.save("models/"+country+"/lstm")
                    else:
                        lstm = keras.models.load_model("models/"+country+"/lstm")
                       
                    
                    if split == 'test':
                        y_pred = normalizer(lstm.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_test = pd.DataFrame(y_test).reset_index(drop=True)
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    #log_params()
                    #log_params({'h_size': elm._h_size, 'activation' :elm.activation_name})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
            

            #ADWIN-SARIMA (AS)
            if model == "AS":
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    if not os.path.exists("models/"+country+"/sarimas"):
                        os.makedirs("models/"+country+"/sarimas")
                    sarimas = {}
                    for k, dft in drift_data.items():
                        try: 
                            sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                        except:
                            try:
                                sarima = train_arima(dft, sarima=True)
                                sarima.save("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                                sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                            except:
                                k -= 1
                        if split == 'val':
                            sarimas[k] = update_arima(sarimas[k], val_data)
                        elif split == 'test':
                            sarimas[k] = update_arima(sarimas[k], test_data)
                        
                    preds = {}
                    for k, m in sarimas.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            preds[k] = m.predict()
                            log_params(m.specification)
                            mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k, 'drift': detector})
                            if split == 'val':
                                metrics = eval_metrics(val_data, preds[k])
                            elif split == 'test':
                                metrics = eval_metrics(test_data.iloc[size:], preds[k].iloc[size:])
                            log_metrics(metrics)
                    pd.DataFrame(preds).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_artifacts("outputs/"+country+"/data")
                    log_artifacts("outputs/"+country+"/preds/"+model)
                    mlflow.end_run()
                
            #ADWIN-SARIMA-ORACLE (ASO)
            if model == "ASO":
                submodel = "AS"
                detector = 'adwin'
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    if split == 'val':
                        true = pd.read_csv("outputs/"+country+"/data/val_data.csv", index_col=0, parse_dates=True)
                    elif split == 'test':
                        true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True)
                    #print(preds)
                    #print(true)
                    #oracle = get_oracle(preds, true)
                    print(preds.shape)
                    print(true.shape)
                    best, oracle = get_oracle(preds, true)
                    pd.Series(best).to_csv("outputs/"+country+"/preds/"+model+"/best.csv")
                    oracle.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size+1:], oracle.iloc[size:])
                    draw_predictions(country, oracle, true)
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size})
                    log_arts(country,model)
                mlflow.end_run()

        #ADWIN-SARIMA-DYNAMIC-SELECTION (ASDS)
            if model == "ASDS":
                submodel = "AS"
                size = int(sys.argv[5])
                K = int(sys.argv[6])
                lags = int(sys.argv[4])
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes, e tente novamente')
                else:
                    sarimas = {}
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    for k, dft in drift_data.items():
                        try: 
                            sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                        except:
                            try:
                                sarima = train_arima(dft, sarima=True)
                                sarima.save("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                                sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                            except:
                                k -= 1
                        if split == 'val':
                            sarimas[k] = update_arima(sarimas[k], val_data)
                            data = val_data
                        elif split == 'test':
                            sarimas[k] = update_arima(sarimas[k], test_data)
                            data = test_data

                    data.index = pd.to_datetime(data.index)
                    preds = {}
                    errors = {}
                    selection = {}
                    for w in data.rolling(window=size):
                        if len(w) == size:
                            #print(w)
                            first = w.index[0]
                            last = w.index[-1]
                            #print(first)
                            #print(last)
                            preds[last] = {}
                            errors[last] = {}
                            selection[last] = {}
                            for k, m in sarimas.items():
                                preds[last][k] = m.predict(start=first, end=last)
                                errors[last][k] = mean_squared_error(preds[last][k], w)
                            #print(preds)
                            #print(errors[last])
                            df_error = pd.Series(errors[last]).rank()
                            #print(df_error)
                            for i in range(K):
                                try:
                                    selection[last][i] = df_error.loc[df_error == i+1].index.values[0]
                                except:
                                    #print(['*']*1000)
                                    #print(df_error.idxmin()) # solucao para ranks malucos 1.5, 2 sem 1...
                                    selection[last][i] = df_error.idxmin()
                            # #print(selection[last])
                            #selection[last] = df_error.loc[df_error < K+1].index.values[:K]
                    df_selection = pd.DataFrame(selection).T
                    df_selection.index = pd.to_datetime(df_selection.index)
                    preds_all = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    preds_selection = {}
                    #print(preds_all)
                    for row in df_selection.iterrows():
                        preds_selection[row[0]] = preds_all.loc[row[0]].iloc[row[1]].mean()
                        #print(row[0])
                        #print(row[1])
                        #print(preds_all.loc[row[0]].iloc[row[1]])
                    preds_selection = pd.Series(preds_selection).T
                    #print(preds_selection)
                    #print(data)
                    #print(data.align(preds_selection, join='right', axis=0))
                    #metrics = eval_metrics(preds_selection, data.reindex_like(preds_selection))
                    metrics = eval_metrics(data.iloc[size:], preds_selection.iloc[1:])
                    draw_predictions(country, preds_selection, data)
                    log_metrics(metrics)
                    df_selection.to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    preds_selection.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_params({'pool': 'sarimas', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size, 'k': k})
                    log_arts(country,model)
                    mlflow.end_run()
            

            #ADWIN-SVM (ASV)
            if model == "ASV":
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    if not os.path.exists("models/"+country+"/svms"):
                        os.makedirs("models/"+country+"/svms")
                    svms = {}
                    for k, dft in drift_data.items():
                        try: 
                            svms[k] = joblib.load("models/"+country+"/svms/svms"+str(k)+".pkl")
                        except:
                            try:
                                X_train, y_train  = splitter(dft, lags)
                                X_val, y_val  = splitter(dft, lags) #gambiarra, é preciso definir o conjunto de validaçao dentro da janela de drift
                                
                                X_train_mm = normalizer(X_train, normx)
                                #y_train = normalizer(y_train.values.reshape(-1,1), normy)
                                X_val_mm = normalizer(X_val, normx)
                                #y_val = normalizer(y_val.values.reshape(-1,1), normy)
                                
                                svm = train_svm(X_train_mm, y_train)
                                joblib.dump(svm, "models/"+country+"/svms/svms"+str(k)+".pkl")
                                svms[k] = joblib.load("models/"+country+"/svms/svms"+str(k)+".pkl")
                            except:
                                k -= 1
                        
                    preds = {}
                    for k, m in svms.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            #y_pred = normalizer(m.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                            y_pred = m.predict(X_test_mm).reshape(-1,1).flatten()
                            #print(y_pred)
                            y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                            preds[k] = y_pred

                            y_test = pd.DataFrame(y_test).reset_index(drop=True)
                            metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])

                            log_params(m.get_params())
                            mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k, 'drift': detector})
                            #metrics = eval_metrics(test_data.iloc[size:], preds[k].iloc[size:])
                            log_metrics(metrics)
                    pd.DataFrame(preds).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_artifacts("outputs/"+country+"/data")
                    log_artifacts("outputs/"+country+"/preds/"+model)
                    mlflow.end_run()

            #ADWIN-SVM-DYNAMIC-SELECTION (ASVDS)
            if model == "ASVDS":
                submodel = "ASV"
                size = int(sys.argv[5])
                K = int(sys.argv[6])
                lags = int(sys.argv[4])
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo ASV antes, e tente novamente')
                else:
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    svms = {}
                    for k, dft in drift_data.items():
                        try: 
                            #print('carrega',k)
                            svms[k] = joblib.load("models/"+country+"/svms/svms"+str(k)+".pkl")
                        except:
                            print('erro carrega', k)
                    
                    #X_test_mm = normalizer(X_test, norms[k])
                    #X_test_mm, y_test = torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
                    #X_test_mm = torcher(X_test_mm)
                    #print('xtest', X_test)
                    #print('ytest', y_test)
                    X_test_mm = normalizer(X_test, normx)
                    preds = {}
                    errors = {}
                    selection = {}
                    for w in range(size,len(y_test)):
                        last = w
                        first = w - size
                        preds[last] = {}
                        errors[last] = {}
                        selection[last] = {}
                        for k, m in svms.items():
                            y_pred = normalizer(m.predict(X_test_mm[first:last]).reshape(-1,1), normy, -1).flatten()
                            #y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                            
                            preds[last][k] = pd.Series(y_pred).reset_index(drop=True)
                            #preds[last][k] = m.predict(X_test_mm[first:last])
                            errors[last][k] = mean_squared_error(y_test.iloc[first:last], preds[last][k])
                        #print(preds)
                        #print(y_test.iloc[first:last])
                        #print(errors[last])
                        df_error = pd.Series(errors[last]).rank()
                        #print(df_error)
                        for i in range(K):
                            try:
                                selection[last][i] = df_error.loc[df_error == i+1].index.values[0]
                            except:
                                #print(['*']*1000)
                                #print(df_error.idxmin()) # solucao para ranks malucos 1.5, 2 sem 1...
                                selection[last][i] = df_error.idxmin()
                        # #print(selection[last])
                        #selection[last] = df_error.loc[df_error < K+1].index.values[:K]
                    df_selection = pd.DataFrame(selection).T
                    #df_selection.index = pd.to_datetime(df_selection.index)
                    preds_all = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    preds_selection = {}
                    #print(preds_all)
                    for row in df_selection.iterrows():
                        preds_selection[row[0]] = preds_all.loc[row[0]].iloc[row[1]].mean()
                        #print(row[0])
                        #print(row[1])
                        #print(preds_all.loc[row[0]].iloc[row[1]])
                    preds_selection = pd.Series(preds_selection).T
                    #print(preds_selection)
                    #print(data)
                    #print(data.align(preds_selection, join='right', axis=0))
                    #metrics = eval_metrics(preds_selection, data.reindex_like(preds_selection))
                    metrics = eval_metrics(pd.DataFrame(y_test).reset_index(drop=True).iloc[size:], preds_selection)
                    draw_predictions(country, preds_selection, data)
                    log_metrics(metrics)
                    pd.DataFrame(errors).to_csv("outputs/"+country+"/preds/"+model+"/errors.csv")
                    pd.DataFrame(preds).to_csv("outputs/"+country+"/preds/"+model+"/preds.csv")
                    df_selection.to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    preds_selection.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size, 'k': k})
                    log_arts(country,model)
                    mlflow.end_run()

            #ADWIN-SVM-ORACLE (ASVO)
            if model == "ASVO":
                submodel = "ASV"
                detector = 'adwin'
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    #print(preds.shape)
                    #print(true.shape)
                    #oracle = get_oracle(preds, true)
                    best, oracle = get_oracle(preds, true)
                    pd.Series(best).to_csv("outputs/"+country+"/preds/"+model+"/best.csv")
                    oracle.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size+1:], oracle.iloc[size:])
                    draw_predictions(country, oracle, true)
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size})
                    log_arts(country,model)
                mlflow.end_run()

            #ADWIN-ELM (AE)
            if model == "AE":
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    if not os.path.exists("models/"+country+"/elms"):
                        os.makedirs("models/"+country+"/elms")
                    if not os.path.exists("models/"+country+"/norms"):
                        os.makedirs("models/"+country+"/norms")
                    elms = {}
                    for k, dft in drift_data.items():
                        try: 
                            elms[k] = torch.load("models/"+country+"/elms/elm"+str(k)+".pkl")
                        except:
                            try:
                                X_train, y_train  = splitter(dft, lags)
                                X_val, y_val  = splitter(dft, lags) #gambiarra, é preciso definir o conjunto de validaçao dentro da janela de drift
                                
                                X_train_mm = normalizer(X_train, normx)
                                y_train_mm = normalizer(y_train.values.reshape(-1,1), normy)
                                X_val_mm = normalizer(X_val, normx)
                                y_val_mm = normalizer(y_val.values.reshape(-1,1), normy)

                                #print('norm')
                                X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                                #print('torch')
                                elm = train_elm(X_train_mm, y_train)
                                #print('treinamento')
                                
                                torch.save(elm ,"models/"+country+"/elms/elm"+str(k)+".pkl")
                                elms[k] = torch.load("models/"+country+"/elms/elm"+str(k)+".pkl")

                            except:
                                pass
                                #k -= 1
                        #joblib.dump(norm, "models/"+country+"/norms/norm"+str(k)+".pkl")

                    preds = {}
                    for k, m in elms.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            X_test_mm = normalizer(X_test, normx)
                            X_test_mm = torcher(X_test_mm)
                            log_params({'h_size': m._h_size, 'activation' :m.activation_name})
                            mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k, 'drift': detector})
                            y_pred = normalizer(m.predict(X_test_mm).numpy().reshape(-1,1), normy, -1).flatten()
                            y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                            #y_pred = post_forecast(pd.Series(m.predict(X_test_mm).numpy().flatten())).reset_index(drop=True)
                            
                            y_test = pd.DataFrame(y_test).reset_index(drop=True)
                            #y_test = pd.DataFrame(y_test.numpy())
                            metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                            log_metrics(metrics)
                            preds[k] = y_pred
                    #print(preds)
                    #print(pd.DataFrame(preds))
                    pd.DataFrame(preds).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_artifacts("outputs/"+country+"/data")
                    log_artifacts("outputs/"+country+"/preds/"+model)
                    mlflow.end_run()

            #ADWIN-ELM-DYNAMIC-SELECTION (AEDS)
            if model == "AEDS":
                submodel = "AE"
                size = int(sys.argv[5])
                K = int(sys.argv[6])
                lags = int(sys.argv[4])
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AE antes, e tente novamente')
                else:
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    elms = {}
                    for k, dft in drift_data.items():
                        try: 
                            #print('carrega',k)
                            elms[k] = torch.load("models/"+country+"/elms/elm"+str(k)+".pkl")
                        except:
                            print('erro carrega', k)
                    
                    #X_test_mm = normalizer(X_test, norms[k])
                    #X_test_mm, y_test = torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
                    #X_test_mm = torcher(X_test_mm)
                    #print('xtest', X_test)
                    #print('ytest', y_test)
                    X_test_mm = normalizer(X_test, normx)
                    X_test_mm = torcher(X_test_mm)
                    preds = {}
                    errors = {}
                    selection = {}
                    for w in range(size,len(y_test)):
                        last = w
                        first = w - size
                        preds[last] = {}
                        errors[last] = {}
                        selection[last] = {}
                        for k, m in elms.items():
                            y_pred = normalizer(m.predict(X_test_mm[first:last]).numpy().reshape(-1,1), normy, -1).flatten()
                            #y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                            
                            preds[last][k] = pd.Series(y_pred).reset_index(drop=True)
                            #preds[last][k] = m.predict(X_test_mm[first:last])
                            errors[last][k] = mean_squared_error(y_test.iloc[first:last], preds[last][k])
                        #print(preds)
                        #print(y_test.iloc[first:last])
                        #print(errors[last])
                        df_error = pd.Series(errors[last]).rank()
                        #print(df_error)
                        for i in range(K):
                            try:
                                selection[last][i] = df_error.loc[df_error == i+1].index.values[0]
                            except:
                                #print(['*']*1000)
                                #print(df_error.idxmin()) # solucao para ranks malucos 1.5, 2 sem 1...
                                selection[last][i] = df_error.idxmin()
                        # #print(selection[last])
                        #selection[last] = df_error.loc[df_error < K+1].index.values[:K]
                    df_selection = pd.DataFrame(selection).T
                    #df_selection.index = pd.to_datetime(df_selection.index)
                    preds_all = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    preds_selection = {}
                    #print(preds_all)
                    for row in df_selection.iterrows():
                        preds_selection[row[0]] = preds_all.loc[row[0]].iloc[row[1]].mean()
                        #print(row[0])
                        #print(row[1])
                        #print(preds_all.loc[row[0]].iloc[row[1]])
                    preds_selection = pd.Series(preds_selection).T
                    #print(preds_selection)
                    #print(data)
                    #print(data.align(preds_selection, join='right', axis=0))
                    #metrics = eval_metrics(preds_selection, data.reindex_like(preds_selection))
                    metrics = eval_metrics(pd.DataFrame(y_test).reset_index(drop=True).iloc[size:], preds_selection)
                    draw_predictions(country, preds_selection, data)
                    log_metrics(metrics)
                    pd.DataFrame(errors).to_csv("outputs/"+country+"/preds/"+model+"/errors.csv")
                    pd.DataFrame(preds).to_csv("outputs/"+country+"/preds/"+model+"/preds.csv")
                    df_selection.to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    preds_selection.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size, 'k': k})
                    log_arts(country,model)
                    mlflow.end_run()

            #ADWIN-ELM-ORACLE (AEO)
            if model == "AEO":
                submodel = "AE"
                detector = 'adwin'
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    #print(preds.shape)
                    #print(true.shape)
                    #oracle = get_oracle(preds, true)
                    best, oracle = get_oracle(preds, true)
                    pd.Series(best).to_csv("outputs/"+country+"/preds/"+model+"/best.csv")
                    oracle.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size+1:], oracle.iloc[size:])
                    draw_predictions(country, oracle, true)
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'size': size})
                    log_arts(country,model)
                mlflow.end_run()

            """
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.statsmodels.log_model(arima, "model", registered_model_name="arimao")
                mlflow.statsmodels.log_model(sarima, "model", registered_model_name="sarimao")
            else:
                mlflow.statsmodels.log_model(arima, "model")
                mlflow.statsmodels.log_model(sarima, "model")
            """
    
if __name__ == "__main__":
    main()   
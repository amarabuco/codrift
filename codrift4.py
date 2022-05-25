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
from utils import *
from charts import *
from drifts import *
from baseline import *
from sarima import *
from elm import *
from lstm import *
from svm import *
from oracle import *

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
#sns.set_theme()

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    
    exp = "refactor"
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

            # subfolders: model e output
            create_dirs(country, model)

            # aquisição de dados
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
                    mlflow.statsmodels.autolog() 
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
                    #log_params(arima.specification)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})
                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
                    
            if model == 'sarima':
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    mlflow.statsmodels.autolog() 
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
                    mlflow.sklearn.autolog()                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train = normalizer(y_train.values.reshape(-1,1), normy).ravel()
                    X_val_mm = normalizer(X_val, normx)
                    y_val = normalizer(y_val.values.reshape(-1,1), normy).ravel()
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
                    mlflow.pytorch.autolog()
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
                    mlflow.keras.autolog()
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val = normalizer(y_val.values.reshape(-1,1), normy)

                    pd.DataFrame(X_train_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_train.csv')
                    pd.DataFrame(y_train).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_train.csv')

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
                        pd.DataFrame(X_test_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_test.csv')
                        y_pred = lstm.predict(X_test_mm)
                        pd.DataFrame(lstm.predict(X_test_mm)).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_test.csv')
                        y_pred = normalizer(y_pred.reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_test = pd.DataFrame(y_test).reset_index(drop=True)
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    #log_params()
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags':lags, 'size': size})
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': k})
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
                                y_train = normalizer(y_train.values.reshape(-1,1), normy).ravel()
                                X_val_mm = normalizer(X_val, normx)
                                y_val = normalizer(y_val.values.reshape(-1,1), normy).ravel()
                                
                                svm = train_svm(X_train_mm, y_train)
                                joblib.dump(svm, "models/"+country+"/svms/svms"+str(k)+".pkl")
                                svms[k] = joblib.load("models/"+country+"/svms/svms"+str(k)+".pkl")
                            except:
                                k -= 1
                        
                    preds = {}
                    for k, m in svms.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            y_pred = normalizer(m.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                            #y_pred = m.predict(X_test_mm).reshape(-1,1).flatten()
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': k})
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags':lags, 'size': size})
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': k})
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
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags':lags, 'size': size})
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
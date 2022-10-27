#Imports

import json
import math
import calendar
import datetime
from datetime import timedelta 
from datetime import datetime as dt
import numpy as np
import pandas as pd
import warnings
import os
import shutil
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
#from mlp import *
from lstm import *
from svm import *
from oracle import *
from ds import *
 

import optuna
import mlflow
import mlflow.statsmodels
from mlflow import log_metric, log_param, log_metrics, log_params, log_artifact, log_artifacts
from urllib.parse import urlparse

from scipy import stats

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
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.optim as optim
#import pytorch_lightning

from keras.backend import clear_session
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from optuna.integration.tensorboard import TensorBoardCallback
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from river import drift

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_theme()
sns.set_theme(style="darkgrid")

import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

def objective_elm(trial):
    #h_size = trial.suggest_int("h_size", 2, 20)
    h_size = trial.suggest_categorical('h_size', [8, 16, 32, 64, 100, 200, 500, 1000])
    activation = trial.suggest_categorical("activation", ["tanh"])
    #activation = trial.suggest_categorical("activation", ["sigmoid", "tanh", "relu", "selu", "elu"])
    init = trial.suggest_categorical("init", ["uniform", "xavier_uniform"])
    #init = trial.suggest_categorical("init", ["uniform", "xavier_uniform", "kaiming_uniform"])
    # Generate the model.
    results = np.zeros(10)
    
    for i in range(10):
        model = ELM(input_size=lags, h_size=h_size, activation=activation, init=init) #variar o input-size (FIX)
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
    best_elm = ELM(input_size=lags, h_size=params['h_size'], activation=params['activation'], init=params['init']) #variar o input-size (FIX)
    best_elm.fit(x, y)
    return best_elm

def objective_elm2(trial):
    #h_size = trial.suggest_int("h_size", 2, 20)
    h_size = trial.suggest_categorical('h_size', [8, 16, 32, 64, 100, 200, 500, 1000])
    activation = trial.suggest_categorical("activation", ["tanh"])
    #activation = trial.suggest_categorical("activation", ["sigmoid", "tanh", "relu", "selu", "elu"])
    init = trial.suggest_categorical("init", ["uniform", "xavier_uniform"])
    #init = trial.suggest_categorical("init", ["uniform", "xavier_uniform", "kaiming_uniform"])
    # Generate the model.
    results = np.zeros(10)
    
    for i in range(10):
        model = ELM(input_size=lags_size, h_size=h_size, activation=activation) #variar o input-size (FIX)
        model.fit(X_train_mm, y_train)
        y_pred = model.predict(X_val_mm)
        mse = -mean_squared_error(y_val, y_pred)
        results[i] = mse

    return results.mean()

def train_elm2(x, y):
    #optimization
    
    study = optuna.create_study(direction="maximize", study_name='elm_study')
    study.optimize(objective_elm2, n_trials=50, show_progress_bar=False)
    params = study.best_params
    best_elm = ELM(input_size=lags_size, h_size=params['h_size'], activation=params['activation']) #variar o input-size (FIX)
    best_elm.fit(x, y)
    return best_elm

def objective_mlp(trial):
    clear_session()
    
    model = Sequential()
    model.add(layers.Dense(units= trial.suggest_categorical('units', [8, 16, 32, 64, 100, 200]), \
        kernel_initializer=trial.suggest_categorical('init', ['he_normal', 'lecun_normal', 'glorot_uniform']), \
        activation=trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'tanh', None]), input_shape=(lags, 1)))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='tanh'))
    #model.add(layers.Dense(1, activation='linear') )
    score = np.zeros(3)
    for i in range(3):
        # We compile our model with a sampled learning rate.
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, decay=1e-4), loss='huber_loss', metrics=['mse'])
        
        log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)        

        model.fit(
            X_train_mm,
            y_train,
            validation_data=(X_val_mm, y_val),
            shuffle=False,
            batch_size=32,
            epochs=100,
            verbose=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), tensorboard_callback]
        )

        # Evaluate the model accuracy on the validation set.
        print(model.evaluate(X_val_mm, y_val, verbose=1))
        _, score[i] = model.evaluate(X_val_mm, y_val, verbose=1)
        #score = model.evaluate(X_val_mm, y_val, verbose=1)
    return -score.mean()   

def mlp(input_size, units, activation, init):
    model = Sequential()
    model.add(layers.Dense(units= units, kernel_initializer=init, activation=activation, input_shape=(input_size, 1)))
    #model.add(layers.Dense(1, activation='linear' ))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='tanh' ))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, decay=1e-4), loss='huber_loss', metrics=['mse'])
    return model

def train_mlp(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='mlp_study')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoardCallback(log_dir, metric_name='mse')
    #earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #study.optimize(objective_mlp, n_trials=10, show_progress_bar=True, callbacks=[ tensorboard_callback])
    study.optimize(objective_mlp, n_trials=10, show_progress_bar=True)
    params = study.best_params
    log_params(study.best_params)
    #best_lstm = lstm(input_size=14, units=params['units'], activation='linear') #variar o input-size (FIX)
    best_mlp = mlp(input_size=lags, units=params['units'], activation=params['activation'], init=params['init']) #variar o input-size (FIX)
    
    log_dir = "logs/fit/best." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    best_mlp.fit(x,
                y,
                validation_data=(X_val_mm, y_val),
                shuffle=False,
                batch_size=32,
                epochs=100,
                verbose=True,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), tensorboard_callback])
    return best_mlp

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():
    
    exp = "AESDS5"
    mlflow.set_experiment(exp)
        
    if sys.argv[1] == 'help':
        print('models: rw, arima, sarima, AS, ASDS, ASO...')
    
    country = sys.argv[1]
    model = sys.argv[2]
    split = sys.argv[3]
    global lags
    lags = int(sys.argv[4])
    size = int(sys.argv[5])
    K = int(sys.argv[6])
    if (model == 'ASDS' or model == 'AEDS' or model == 'AEDS3' or model == 'AEDS4' or model == 'AEDS5' or model == 'AEDS2' or model == 'ASVDS'):
        rname = country+'.'+model+'.'+split+'.'+str(lags)+'.'+str(size)+'.'+str(K)
    else:
        rname =  country+'.'+model+'.'+split
    if (model == 'AEDS5' or model == 'AEDS' or model == 'ASDS'):
        mtr = sys.argv[7]

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

            normx, X_train_mm = normalizer(X_train)
            normy, y_train_mm = normalizer(y_train.values.reshape(-1,1))

            X_val_mm = normalizer(X_val, normx)
            X_test_mm = normalizer(X_test, normx)
            y_val_mm = normalizer(y_val.values.reshape(-1,1), normy)
            y_test_mm = normalizer(y_test.values.reshape(-1,1), normy)

            norm2, _ = normalizer2(y_train.diff(1).dropna().values.reshape(-1,1)) # diff normalization

            if model == "detector":
                #params = {'ws': [30,60,90,120], 'sz' : [7,14,28], 'alpha' : [0.005, 0.01, 0.025]}
                params = {'ws': [60,120], 'sz' : [7,14,28], 'alpha' : [0.005, 0.01]}
                for p in ParameterGrid(params):
                    with mlflow.start_run(run_name=country+'.'+split, nested=True):
                        detector = split
                        if not os.path.exists("outputs/"+country+"/drifts/"+detector):
                            os.makedirs("outputs/"+country+"/drifts/"+detector)
                        detector_model = set_detector(detector, window_size=p['ws'], stat_size= p['sz'], alpha=p['alpha'])
                        drifts, drift_data, drift_detector = get_drifts2(train_data, country, detector_model)
                        df_drift = pd.DataFrame(drift_data.values(), index=drift_data.keys())
                        df_drift.to_csv("outputs/"+country+"/drifts/"+detector+"/drifts.csv")
                        drift_size = np.array([len(v) for v in drift_data.values()])
                        for k, v in drift_data.items():
                            with mlflow.start_run(run_name=country+'.'+split+'.'+str(k), nested=True):
                                g = stats.linregress(v.reset_index(drop=True).reset_index().to_numpy())
                                log_metric('slope', g[0])
                                log_metric('intercept', g[1])
                                log_metric('mean', np.mean(v).to_numpy()[0])
                                log_metric('std', np.std(v).to_numpy()[0])
                                log_metric('1q', v.quantile(0.25).to_numpy()[0])
                                log_metric('2q', np.median(v))
                                log_metric('3q', v.quantile(0.75).to_numpy()[0])
                                mlflow.set_tags({'data': country,  'detector': detector, 'split': str(k)})
                        log_metrics({'drifts': len(drift_data), 'min': drift_size.min(), 'median': np.median(drift_size), 'max': drift_size.max()})
                        log_artifact(f"outputs/{country}/drifts/{detector}/drifts.csv")
                        draw_drifts(country, drifts, drift_data, train_data)
                        log_artifact(f"outputs/{country}/drifts.png")
                        log_params(drift_detector)
                        mlflow.set_tags({'data': country,  'detector': detector})
                        mlflow.end_run()
                mlflow.end_run()
                    
            print(model)

            if model == 'rw':
                    print('rw')
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
                    y_val_org = y_val.copy()
                    y_val = normalizer(y_val.values.reshape(-1,1), normy).ravel()
                    X_test_mm = normalizer(X_test, normx)

                    #print(X_train_mm)
                    #print(y_train)

                    y_val_org.to_csv("outputs/"+country+"/data/y_val.csv")
                    y_test.to_csv("outputs/"+country+"/data/y_test.csv")
                    
                    #print(np.concatenate((X_train_mm, X_val_mm), axis=0), np.concatenate((y_train, y_val), axis=0))
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/svm.pkl"):
                        svm = train_svm(X_train_mm, y_train)
                        joblib.dump(svm, "models/"+country+"/svm.pkl") 
                    else:
                        svm = joblib.load("models/"+country+"/svm.pkl")                  
                    
                    if split == 'val':
                        #y_pred = normalizer(svm.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                        y_pred = svm.predict(X_val_mm).reshape(-1,1).flatten()
                        y_pred = post_forecast(pd.DataFrame(y_pred))
                        #y_pred = post_forecast(pd.DataFrame(svm.predict(X_test_mm)))
                        y_val = pd.DataFrame(y_val_org).reset_index(drop=True) #y_val está normalizado, então precisa pegar o valor original
                        metrics = eval_metrics(y_val.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_val)
      
                    if split == 'test':
                        #y_pred = normalizer(svm.predict(X_test_mm).reshape(-1,1), normy, -1).flatten()
                        y_pred = svm.predict(X_test_mm).reshape(-1,1).flatten()
                        y_pred = post_forecast(pd.DataFrame(y_pred))
                        #y_pred = post_forecast(pd.DataFrame(svm.predict(X_test_mm)))
                        y_test = pd.DataFrame(y_test).reset_index(drop=True)
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    log_params(svm.get_params())
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()
                
            if model == "elm":
                    #mlflow.pytorch.autolog()
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train_mm = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val_org = y_val.copy()
                    y_val_mm = normalizer(y_val.values.reshape(-1,1), normy)

                    X_test_mm = normalizer(X_test, normx)
                    X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train.values.reshape(-1,1)), torcher(X_val_mm), torcher(y_val.values.reshape(-1,1))
                    X_test_mm, y_test= torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
        
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if os.path.exists("models/"+country+"/elm.pkl"):
                        elm = train_elm(X_train_mm, y_train)
                        torch.save(elm ,"models/"+country+"/elm.pkl")
                        elm = torch.load("models/"+country+"/elm.pkl")
                    else:
                        elm = train_elm(X_train_mm, y_train)
                        torch.save(elm ,"models/"+country+"/elm.pkl")
                    
                    if split == 'val':
                        y_pred = normalizer(elm.predict(X_val_mm).numpy().reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_val = pd.DataFrame(y_val_org).reset_index(drop=True)
                        metrics = eval_metrics(y_val.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_val)
                    
                    if split == 'test':
                        y_pred = normalizer(elm.predict(X_test_mm).numpy().reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_test = pd.DataFrame(y_test.numpy()).reset_index(drop=True)
                        metrics = eval_metrics(y_test.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_test)
                    log_metrics(metrics)
                    log_params({'h_size': elm._h_size, 'activation': elm.activation_name, 'init': elm.init})
                    #log_params({'h_size': elm._h_size, 'activation' :elm.activation_name})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'size': size})

                    pd.DataFrame(y_pred).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_arts(country,model)
                    mlflow.end_run()

            if model == 'mlp':
                    shutil.rmtree("logs/")
                    if not os.path.exists("logs/"):
                        os.makedirs("logs/fit/")
                    mlflow.tensorflow.autolog()
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val_org = y_val.copy()
                    y_val = normalizer(y_val.values.reshape(-1,1), normy)

                    pd.DataFrame(X_train_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_train.csv')
                    pd.DataFrame(y_train).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_train.csv')

                    X_test_mm = normalizer(X_test, normx)
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                    #X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train.values.reshape(-1,1)), torcher(X_val_mm), torcher(y_val.values.reshape(-1,1))
                    #X_test_mm, y_test= torcher(X_test_mm), torcher(y_test.values.reshape(-1,1))
        
                #with mlflow.start_run(run_name=country+'.'+model+'.'+split):
                    if not os.path.exists("models/"+country+"/mlp"):
                        mlp = train_mlp(X_train_mm, y_train)
                        mlp.save("models/"+country+"/mlp")
                    else:
                        mlp = train_mlp(X_train_mm, y_train)
                        mlp.save("models/"+country+"/mlp")
                        #mlp = keras.models.load_model("models/"+country+"/mlp")

                    if split == 'val':
                        pd.DataFrame(X_val_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_test.csv')
                        print('train', mlp.predict(X_train_mm).shape)
                        print('test', mlp.predict(X_val_mm).shape)
                        y_pred = mlp.predict(X_val_mm)
                        mlp.predict(X_val_mm)
                        #pd.DataFrame(mlp.predict(X_test_mm.reshape(-1,14))).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_test.csv')
                        y_pred = normalizer(y_pred.reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_val = pd.DataFrame(y_val_org).reset_index(drop=True)
                        metrics = eval_metrics(y_val.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_val)
             
                    if split == 'test':
                        pd.DataFrame(X_test_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_test.csv')
                        print('train', mlp.predict(X_train_mm).shape)
                        print('test', mlp.predict(X_test_mm).shape)
                        y_pred = mlp(X_test_mm, training=False)
                        mlp.predict(X_test_mm)
                        #pd.DataFrame(mlp.predict(X_test_mm.reshape(-1,14))).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_test.csv')
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
            
            
            if model == 'lstm':
                    mlflow.keras.autolog()
                    #X_train_mm, y_train, X_val_mm, y_val, X_test_mm, y_test = torch_data(train_data, val_data, test_data, lags)
                    
                    #norm, X_train_mm = normalizer(X_train)
                    #X_val_mm = normalizer(X_val, norm)

                    X_train_mm = normalizer(X_train, normx)
                    y_train = normalizer(y_train.values.reshape(-1,1), normy)
                    X_val_mm = normalizer(X_val, normx)
                    y_val_org = y_val.copy()
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
                    
                    if split == 'val':
                        pd.DataFrame(X_val_mm).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_input_test.csv')
                        y_pred = lstm.predict(X_val_mm)
                        pd.DataFrame(lstm.predict(X_val_mm)).to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_output_test.csv')
                        y_pred = normalizer(y_pred.reshape(-1,1), normy, -1).flatten()
                        y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                        #y_pred = post_forecast(pd.Series(elm.predict(X_test_mm).numpy().flatten()))
                        y_val = pd.DataFrame(y_val_org).reset_index(drop=True)
                        metrics = eval_metrics(y_val.iloc[7:], y_pred.iloc[7:])
                        draw_predictions(country, y_pred, y_val)
           
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
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                #pd.DataFrame(drift_data).to_csv("outputs/"+country+"/data/"+model+"/drifts.csv")
                #log_artifact('outputs/'+country+'/drifts.png')
                #log_artifact('outputs/'+country+'/data/'+model+'/drifts.csv')
                mlflow.set_tags({'drifts': len(drifts)})
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
                    metrics = eval_metrics(true.iloc[size+1:].reset_index(drop=True), oracle.iloc[size:].reset_index(drop=True))
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
                                #errors[last][k] = mean_squared_error(preds[last][k], w)
                                errors[last][k] = get_mtr(mtr, preds[last][k], w)
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
                    log_params({'pool': 'sarimas', 'window_size': size ,'K':K, 'metric':mtr, 'distance': None})
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
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                log_artifact('outputs/'+country+'/drifts.png')
                mlflow.set_tags({'drifts': len(drifts)})
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
                            1 + '1' #gera erro
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

            #ADWIN-ELM (AE3)
            if model == "AE3":
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    if not os.path.exists("models/"+country+"/elms3"):
                        os.makedirs("models/"+country+"/elms3")
                    if not os.path.exists("models/"+country+"/norms"):
                        os.makedirs("models/"+country+"/norms")
                    elms = {}
                    lags_params = {}
                    for k, dft in drift_data.items():
                        try:
                            lags_selection = select_lags(dft, lags)
                            mlflow.set_tag('lags',lags)
                            alpha = 0.05
                            if len(lags_selection) == 0:
                                alpha = 0.1
                                lags_selection = select_lags(dft, lags, alpha=alpha)
                            mlflow.set_tag('alpha',alpha)
                            lags_params[k] = lags_selection
                            print(lags_selection)
                            
                            global lags_size
                            lags_size = len(lags_selection)
                            X_train, y_train  = splitter(dft, lags)
                            X_train = get_lags(X_train, lags_selection)
                            X_val, y_val  = splitter(dft, lags) #gambiarra, é preciso definir o conjunto de validaçao dentro da janela de drift
                            X_val = get_lags(X_val, lags_selection)
                            
                            print(y_train.shape)
                            norm, y_train_mm = normalizer2(y_train.values.reshape(-1,1))
                            print(X_train.shape)
                            X_train_mm = normalizer2(X_train, norm)
                            y_val_mm = normalizer2(y_val.values.reshape(-1,1), norm)
                            X_val_mm = normalizer2(X_val, norm)
                                                        
                            #print('norm')
                            X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                            #print('torch')
                            elm = train_elm2(X_train_mm, y_train)
                            #print('treinamento')
                            
                            torch.save(elm ,"models/"+country+"/elms3/elm"+str(k)+".pkl")
                            elms[k] = torch.load("models/"+country+"/elms3/elm"+str(k)+".pkl")
                            """
                            try: 
                                1 + '1' #gera erro
                                elms[k] = torch.load("models/"+country+"/elms/elm"+str(k)+".pkl")
                            except:
                                try:
                                    lags_selection = select_lags(dft)
                                    print(lags_selection)                          
                                    X_train, y_train  = splitter(dft, lags)
                                    X_train = get_lags(X_train, lags_selection)
                                    X_val, y_val  = splitter(dft, lags) #gambiarra, é preciso definir o conjunto de validaçao dentro da janela de drift
                                    X_val = get_lags(X_val, lags_selection)
                                    
                                    norm, y_train_mm = normalizer2(y_train)
                                    X_train_mm = normalizer2(X_train, norm)
                                    y_val_mm = normalizer2(y_val, norm)
                                    X_val_mm = normalizer2(X_val, norm)
                                    
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
                        """
                        except:
                            print(k, len(dft), 'erro')
                    
                    preds = {}
                    for k, m in elms.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            print(k, lags_params[k])
                            X_test_mm = get_lags(X_test, lags_params[k])
                            X_test_mm = normalizer2(X_test_mm.values, norm)
                            X_test_mm = torcher(X_test_mm)
                            log_params({'h_size': m._h_size, 'activation' :m.activation_name, 'lags_selection': lags_params[k]})
                            mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k, 'drift': detector})
                            y_pred = normalizer2(m.predict(X_test_mm).numpy().reshape(-1,1), norm, -1).flatten()
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

            #ADWIN-ELM (AE4)
            if model == "AE4":
                    detector = 'adwin'
                    drifts, drift_data = get_drifts(train_data, country, detector=detector)
                    draw_drifts(country, drifts, drift_data, train_data)
                    log_artifact('outputs/'+country+'/drifts.png')
                    if not os.path.exists("models/"+country+"/elms4"):
                        os.makedirs("models/"+country+"/elms4")
                    if not os.path.exists("models/"+country+"/norms"):
                        os.makedirs("models/"+country+"/norms")
                    elms = {}
                    diffs = {}
                    adf = {}
                    for k, dft in drift_data.items():
                        try: 
                            1 + '1' #gera erro
                            elms[k] = torch.load("models/"+country+"/elms4/elm"+str(k)+".pkl")
                        except:
                            try:
                                dft = dft.diff(1).dropna() # DIFF
                                diffs[k] = dft
                                adf[k] = tsa.stattools.adfuller(dft, maxlag=lags)[1]
                                X_train, y_train  = splitter(dft, lags)
                                X_val, y_val  = splitter(dft, lags) #gambiarra, é preciso definir o conjunto de validaçao dentro da janela de drift
                                
                                print(y_train.shape)
                                y_train_mm = normalizer2(y_train.values.reshape(-1,1), norm2)
                                print(X_train.shape)
                                X_train_mm = normalizer2(X_train, norm2)
                                y_val_mm = normalizer2(y_val.values.reshape(-1,1), norm2)
                                X_val_mm = normalizer2(X_val, norm2)

                                #print('norm')
                                X_train_mm, y_train, X_val_mm, y_val = torcher(X_train_mm), torcher(y_train_mm.reshape(-1,1)), torcher(X_val_mm), torcher(y_val_mm.reshape(-1,1))
                                #print('torch')
                                elm = train_elm(X_train_mm, y_train)
                                #print('treinamento')
                                
                                torch.save(elm ,"models/"+country+"/elms4/elm"+str(k)+".pkl")
                                elms[k] = torch.load("models/"+country+"/elms4/elm"+str(k)+".pkl")
                                

                            except:
                                print('erro treinamento')
                                #pass
                                #k -= 1
                        #joblib.dump(norm, "models/"+country+"/norms/norm"+str(k)+".pkl")
                    
                    draw_diffs(country, diffs)
                    log_artifact('outputs/'+country+'/diffs.png')
                    pd.Series(adf).to_csv("outputs/"+country+"/data/adf.csv")
                    preds = {}
                    X_test.to_csv("outputs/"+country+"/data/xtest.csv")
                    X_test = X_test.diff(1).dropna()
                    #print('base', X_test.shape)
                    X_test.to_csv("outputs/"+country+"/data/xtest_diff.csv")
                    #print('diff', X_test.shape)
                    X_test_mm = normalizer2(X_test, norm2)
                    np.savetxt("outputs/"+country+"/data/xtest_norm.csv", X_test_mm, delimiter=',')
                    X_test_mm = torcher(X_test_mm)
                    for k, m in elms.items():
                        with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                            log_params({'h_size': m._h_size, 'activation' :m.activation_name, 'init' :m.init})
                            mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k, 'drift': detector})
                            y_pred = normalizer2(m.predict(X_test_mm).numpy().reshape(-1,1), norm2, -1).flatten()
                            #print('pre_cum', y_pred)
                            #print(y_pred.shape)
                            y_pred = cum_forecast(y_test.reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)) # CUM
                            #print(y_test.shape)
                            #print('cum_forecast', y_pred)
                            y_pred = post_forecast(y_pred).reset_index(drop=True)
                            #print(y_pred)
                            
                            
                            y_test_tmp = pd.DataFrame(y_test).iloc[1:].reset_index(drop=True)
                            #print(y_test.iloc[8:], y_pred.iloc[7:])
                            #y_test = pd.DataFrame(y_test.numpy())
                            #print(y_test)
                            #print(y_pred.iloc[7:])
                            metrics = eval_metrics(y_test_tmp.iloc[7:], y_pred.iloc[7:])
                            log_metrics(metrics)
                            preds[k] = y_pred
                    #print(preds)
                    print(pd.DataFrame(preds))
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
                if not os.path.exists("datax/"+country+"/"+model):
                        os.makedirs("datax/"+country+"/"+model)
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
                    #pd.DataFrame(X_test).to_csv("datax/"+country+"/"+model+"/x_test.csv")
                    X_test_mm = torcher(X_test_mm)
                    #pd.DataFrame(X_test_mm).to_csv("datax/"+country+"/"+model+"/x_test_torch.csv")
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
                            pd.DataFrame(X_test[first:last]).to_csv("datax/"+country+"/"+model+"/x_test_"+str(w)+"_"+str(k)+".csv")
                            y_pred = normalizer(m.predict(X_test_mm[first:last]).numpy().reshape(-1,1), normy, -1).flatten()
                            y_pred = post_forecast(pd.Series(y_pred)).reset_index(drop=True)
                            #pd.DataFrame(y_pred).to_csv("datax/"+country+"/"+model+"/y_test_"+str(w)+"_"+str(k)+".csv")
                            
                            preds[last][k] = pd.Series(y_pred).reset_index(drop=True)
                            #preds[last][k] = m.predict(X_test_mm[first:last])
                            pd.DataFrame(y_test.iloc[first:last]).to_csv("datax/"+country+"/"+model+"/true_"+str(w)+"_"+str(k)+".csv")
                            #errors[last][k] = mean_squared_error(y_test.iloc[first:last].reset_index(drop=True), preds[last][k])                            
                            errors[last][k] = get_mtr(mtr, y_test.iloc[first:last].reset_index(drop=True), preds[last][k])                            
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
                    #print(data)
                    #print(data.align(preds_selection, join='right', axis=0))
                    #metrics = eval_metrics(preds_selection, data.reindex_like(preds_selection))
                    metrics = eval_metrics(pd.DataFrame(y_test).reset_index(drop=True).iloc[size:], preds_selection)
                    #print(metrics)
                    draw_predictions(country, preds_selection, test_data)
                    log_metrics(metrics)
                    pd.DataFrame(errors).to_csv("outputs/"+country+"/preds/"+model+"/errors.csv")
                    #print(preds)
                    preds_df = pd.DataFrame()
                    for w, pred in enumerate(preds.values()):
                        tmp = pd.DataFrame(pred).T
                        tmp['window'] = w
                        preds_df = preds_df.append(tmp)
                    pd.DataFrame(preds_df).to_csv("outputs/"+country+"/preds/"+model+"/preds.csv")
                    df_selection.to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    preds_selection.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':mtr, 'distance': None})
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': k})
                    #### ORACLE ####
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    best, oracle = get_oracle(preds, true)
                    pd.Series(best).to_csv("outputs/"+country+"/preds/"+model+"/best.csv")
                    oracle.to_csv("outputs/"+country+"/preds/"+model+"/oracle.csv")
                    draw_predictions(country, preds_selection, oracle, True)
                    log_arts(country,model)                                        
                    mlflow.end_run()

            #ADWIN-ELM-ORACLE (AEDS3)
            if model == "AEDS3":
                submodel = "AE3"
                detector = "adwin"
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                dfts = drifts2df(drift_data)
                dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                log_artifact('outputs/'+country+'/drifts.png')
                log_artifact('outputs/'+country+'/data/drifts.csv')
                mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    #print(preds.shape)
                    #print(true.shape)
                    #oracle = get_oracle(preds, true)
                    selection, selection_preds = get_ds(preds, true, size, K)
                    pd.Series(selection).to_csv("outputs/"+country+"/preds/"+model+"/best.csv")
                    selection_preds.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size:].reset_index(drop=True), selection_preds)
                    draw_predictions(country, selection_preds, true)
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': k})
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    log_arts(country,model)
                mlflow.end_run()
                
            #ADWIN-ELM-ORACLE (AEDS4)
            if model == "AEDS4":
                submodel = "AE4"
                detector = "adwin"
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                dfts = drifts2df(drift_data)
                dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                log_artifact('outputs/'+country+'/drifts.png')
                log_artifact('outputs/'+country+'/data/drifts.csv')
                log_artifact('outputs/'+country+'/diffs.png')
                mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    #print(preds.shape)
                    #print(true.shape)
                    #oracle = get_oracle(preds, true)
                    selection, selection_preds = get_ds(preds, true, size, K)
                    pd.Series(selection).to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    selection_preds.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size+1:].reset_index(drop=True), selection_preds)
                    draw_predictions(country, selection_preds, true.iloc[size+1:].reset_index(drop=True))
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': K})
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    log_arts(country,model)
                mlflow.end_run()

            #ADWIN-ELM-ORACLE (AEDS5)
            if model == "AEDS5":
                #mtr = get_mtr(mtr)
                submodel = "AE4"
                detector = "adwin"
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                dfts = drifts2df(drift_data)
                dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                log_artifact('outputs/'+country+'/drifts.png')
                log_artifact('outputs/'+country+'/data/drifts.csv')
                log_artifact('outputs/'+country+'/diffs.png')
                mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    #print(preds.shape)
                    #print(true.shape)
                    #oracle = get_oracle(preds, true)
                    selection, selection_preds = get_ds2(preds, true, mtr, size, K)
                    pd.Series(selection).to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    selection_preds.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size+1:].reset_index(drop=True), selection_preds)
                    draw_predictions(country, selection_preds, true.iloc[size+1:].reset_index(drop=True))
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': K})
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':mtr, 'distance': None})
                    log_arts(country,model)
                mlflow.end_run()

            if model == "AEO":
                submodel = "AE4"
                detector = 'adwin'
                drifts, drift_data = get_drifts(train_data, country, detector=detector)
                draw_drifts(country, drifts, drift_data, train_data)
                dfts = drifts2df(drift_data)
                dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                log_artifact('outputs/'+country+'/diffs.png')
                #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                log_artifact('outputs/'+country+'/drifts.png')
                log_artifact('outputs/'+country+'/data/drifts.csv')
                mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+submodel):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds = pd.read_csv("outputs/"+country+"/preds/"+submodel+"/"+submodel+'_'+split+'.csv', index_col=0, parse_dates=True)
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, parse_dates=True).reset_index(drop=True)
                    print(preds.shape)
                    print(true.shape)
                    #oracle = get_oracle(preds, true)
                    if submodel == "AE4":
                        best, oracle = get_oracle(preds, true.iloc[1:])
                    else:
                        best, oracle = get_oracle(preds, true)
                    pd.Series(best).to_csv("outputs/"+country+"/preds/"+model+"/oracle.csv")
                    oracle.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    if submodel == "AE4":
                        metrics = eval_metrics(true.iloc[2:].iloc[size:].reset_index(drop=True), oracle.iloc[size:].reset_index(drop=True))
                        draw_predictions(country, oracle.iloc[size:].reset_index(drop=True), true.iloc[2:].iloc[size:].reset_index(drop=True))
                    else:
                        metrics = eval_metrics(true.iloc[size+1:].reset_index(drop=True), oracle.iloc[size:].reset_index(drop=True))
                        draw_predictions(country, oracle.iloc[size:].reset_index(drop=True), true.iloc[size+1:].reset_index(drop=True))
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags':lags, 'size': size})
                    log_arts(country,model)
                mlflow.end_run()

            #ADWIN-ELM-ORACLE (AESDS)
            if model == "AESDS":
                submodel1 = "AS"
                submodel2 = "AE4"
                detector = 'adwin'
                # drifts, drift_data = get_drifts(train_data, country, detector=detector)
                # draw_drifts(country, drifts, drift_data, train_data)
                # dfts = drifts2df(drift_data)
                # dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                # log_artifact('outputs/'+country+'/diffs.png')
                # #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                # log_artifact('outputs/'+country+'/drifts.png')
                # log_artifact('outputs/'+country+'/data/drifts.csv')
                # mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+model):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    preds1 = pd.read_csv("outputs/"+country+"/preds/"+submodel1+"/"+submodel1+'_'+split+'.csv', index_col=0, parse_dates=True)
                    preds2 = pd.read_csv("outputs/"+country+"/preds/"+submodel2+"/"+submodel2+'_'+split+'.csv', index_col=0, parse_dates=True)
                    preds = pd.concat([preds1.iloc[1:].reset_index(drop=True), preds2], axis=1)
                    preds.columns = list(range(preds.shape[-1]))
                    true = pd.read_csv("outputs/"+country+"/data/test_data.csv", index_col=0, skiprows=1, parse_dates=True).reset_index(drop=True)
                    #print(preds, true.iloc[1:])
                    selection, selection_preds = get_ds(preds, true, size, K)
                    #print(selection_preds)
                    pd.Series(selection).to_csv("outputs/"+country+"/preds/"+model+"/selection.csv")
                    selection_preds.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(true.iloc[size:].reset_index(drop=True), selection_preds)
                    draw_predictions(country, selection_preds, true.iloc[size:].reset_index(drop=True))
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': K})
                    log_params({'pool': 'elms', 'window_size': size ,'K':K, 'metric':'mse', 'distance': None})
                    log_arts(country,model)
                mlflow.end_run()

            #ENSEMBLE (ENS)
            if model == "ensemble":
                submodel1 = "arima"
                submodel2 = "sarima"
                submodel3 = "elm"
                submodel4 = "svm"
                submodel5 = "lstm"
                submodel6 = "mlp"
                detector = 'adwin'
                #submodels = [submodel1, submodel2, submodel3, submodel4, submodel5]
                submodels = [submodel1, submodel2, submodel3, submodel4]
                # drifts, drift_data = get_drifts(train_data, country, detector=detector)
                # draw_drifts(country, drifts, drift_data, train_data)
                # dfts = drifts2df(drift_data)
                # dfts.to_csv("outputs/"+country+"/data/drifts.csv")
                # log_artifact('outputs/'+country+'/diffs.png')
                # #pd.DataFrame(drift_data.values()).to_csv("outputs/"+country+"/data/drifts.csv")
                # log_artifact('outputs/'+country+'/drifts.png')
                # log_artifact('outputs/'+country+'/data/drifts.csv')
                # mlflow.set_tags({'drifts': len(drifts)})
                if not os.path.exists("outputs/"+country+"/preds/"+model):
                    print('execute o modelo AS antes e depois tente novamente')
                else:
                    from sklearn.tree import DecisionTreeRegressor
                    stk_model = DecisionTreeRegressor(random_state=0) # falta otimizar
                    X_vals = []
                    for sub in submodels:
                        tmp = pd.read_csv("outputs/"+country+"/preds/"+sub+"/"+sub+'_val.csv', index_col=0, parse_dates=True)
                        if sub in ['arima', 'sarima']:
                            tmp = tmp.iloc[5:]
                        if sub in ['svm', 'elm', 'lstm', 'mlp']:
                            tmp = tmp.iloc[:-2]
                        X_vals.append(tmp.reset_index(drop=True))                            
                    X_val = pd.concat(X_vals, axis=1)
                    # print(X_val)
                    # print(y_val)
                    stk_model.fit(X_val, y_val.iloc[:-2])
                    X_tests = []
                    for sub in submodels:
                        tmp = pd.read_csv("outputs/"+country+"/preds/"+sub+"/"+sub+'_test.csv', index_col=0, parse_dates=True).reset_index(drop=True)
                        # if sub in ['arima', 'sarima']:
                        #     tmp = tmp.iloc[5:]
                        # if sub in ['svm', 'elm', 'lstm', 'mlp']:
                        #     tmp = tmp.iloc[:-5]
                        X_tests.append(tmp)
                    X_test = pd.concat(X_tests, axis=1)
                    pred = pd.DataFrame(stk_model.predict(X_test))
                    pred = X_test.median(axis=1)
                    #pred = X_test.mean(axis=1)
                    # print(pred.shape)
                    # print(y_test.shape)
                    pred.to_csv("outputs/"+country+"/preds/"+model+"/"+model+'_'+split+'.csv')
                    metrics = eval_metrics(test_data.iloc[-65:].reset_index(drop=True), pred.iloc[-65:].reset_index(drop=True))
                    draw_predictions(country, pred, test_data.iloc[size:].reset_index(drop=True))
                    log_metrics(metrics)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'drift': detector, 'lags': lags, 'size': size, 'k': K})
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

         
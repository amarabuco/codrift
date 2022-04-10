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
warnings.filterwarnings('ignore')

import mlflow
import mlflow.statsmodels
from mlflow import log_metric, log_param, log_metrics, log_params, log_artifact, log_artifacts
from urllib.parse import urlparse

from pmdarima.arima import auto_arima as pmautoarima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from river import drift

#visualization
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_theme()


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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
    
    #return data_diff


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

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2':r2}

def main():
    
    mlflow.set_experiment("codrift")
    
    if sys.argv[1] == 'help':
        print('models: rw, arima, sarima, AS, ASDS, ASO...')
    
    country = sys.argv[1]
    model = sys.argv[2]
    split = sys.argv[3]
    
    data = get_data('covid', country+'_daily.csv')
    data = fix_outliers(data)
    train_data, val_data, test_data = train_val_test_split(data, 0.7, 0.2)
    print('train: ', train_data.head(1), train_data.tail(1), len(train_data) )
    print('val: ', val_data.head(1), val_data.tail(1), len(val_data) )
    print('test: ', test_data.head(1), test_data.tail(1), len(test_data) )
    
    if not os.path.exists("outputs/"+country+"/data"):
        os.makedirs("outputs/"+country+"/data")
    
    train_data.to_csv("outputs/"+country+"/data/train.csv")
    val_data.to_csv("outputs/"+country+"/data/val_data.csv")
    test_data.to_csv("outputs/"+country+"/data/test_data.csv")
    
    
   
    """
    fig, ax = plt.subplots()
    ax.plot(train_data, color='black')
    ax.plot(val_data, color='darkgray')
    ax.plot(test_data, color='lightgray')
    plt.savefig('outputs/'+country+'/data.png')
   
    # Log an artifact (output file)
    log_artifacts('outputs/'+country+'/data.png')
    """
    
    # Drifts
    
    #print(data.head())
    
    #print(drifts)

    if not os.path.exists("models/"+country):
        os.makedirs("models/"+country)

    if model == "AS":
        with mlflow.start_run(run_name=country+'.'+model+'.'+split):
            drifts, drift_data = get_drifts(train_data, country, detector='adwin')
            if not os.path.exists("models/"+country+"/sarimas"):
                os.makedirs("models/"+country+"/sarimas")
            sarimas = {}
            for k, dft in drift_data.items():
                try: 
                    sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                except:
                    sarima = train_arima(dft, sarima=True)
                    sarima.save("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                    sarimas[k] = sm.load("models/"+country+"/sarimas/sarima"+str(k)+".pkl")
                if split == 'val':
                    sarimas[k] = update_arima(sarimas[k], val_data)
                elif split == 'test':
                    sarimas[k] = update_arima(sarimas[k], test_data)
                
            preds = {}
            for k, m in sarimas.items():
                with mlflow.start_run(run_name=country+'.'+model+'.'+split+'.'+str(k), nested=True):
                    preds[k] = m.predict()
                    log_params(m.specification)
                    mlflow.set_tags({'data': country, 'split': split, 'model': model, 'submodel': k})
                    if split == 'val':
                        metrics = eval_metrics(preds[k], val_data)
                    elif split == 'test':
                        metrics = eval_metrics(preds[k], test_data)
                    log_metrics(metrics)    
            log_artifacts("outputs/"+country+"/data")
                    
    if model == 'rw':
        with mlflow.start_run(run_name=country+'.'+model+'.'+split):
        #Baseline 
            if split == 'val':
                metrics = eval_metrics(random_walk(val_data), test_data[1:])
            elif split == 'test':
                metrics = eval_metrics(random_walk(test_data), test_data[1:])
            mlflow.set_tags({'data': country, 'split': split, 'model': model})
            log_metrics(metrics)
            log_artifacts("outputs/"+country+"/data")
            
    if model == 'arima':
        with mlflow.start_run(run_name=country+'.'+model+'.'+split):
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
                y_pred = arima.predict()
                metrics = eval_metrics(y_pred, val_data)
            elif split == 'test':
                arima = update_arima(arima, test_data)
                y_pred = arima.predict()
                metrics = eval_metrics(y_pred, test_data)
            log_metrics(metrics)
            log_params(arima.specification)
            mlflow.set_tags({'data': country, 'split': split, 'model': model})
            log_artifacts("outputs/"+country+"/data")
            
    if model == 'sarima':
        with mlflow.start_run(run_name=country+'.'+model+'.'+split):
            if not os.path.exists("models/"+country+"/sarima.pkl"):
                sarima = train_arima(train_data, sarima=True)
                sarima.save("models/"+country+"/sarima.pkl")
            else:
                sarima = sm.load("models/"+country+"/sarima.pkl")
            log_params(sarima.specification)
            
            if split == 'val':
                sarima = update_arima(sarima, val_data)
                y_pred = sarima.predict()
                metrics = eval_metrics(y_pred, test_data)
            elif split == 'test':
                sarima = update_arima(sarima, test_data)
                y_pred = sarima.predict()
                metrics = eval_metrics(y_pred, test_data)
            log_metrics(metrics)
            log_params(sarima.specification)
            mlflow.set_tags({'data': country, 'split': 'val', 'model': 'sarima'})

            with open("outputs/"+country+"/sarima.txt", "w") as f:
                f.write(sarima.summary().as_text())
            log_artifact("outputs/"+country+"/sarima.txt")
            log_artifacts("outputs/"+country+"/data")
        
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
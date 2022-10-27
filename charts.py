import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_data(country, train,val,test):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train, color='black', label='train')
    ax.plot(val, color='darkgray', label='val')
    ax.plot(test, color='lightgray', label='test')
    plt.xticks([])
    plt.title(country)
    plt.legend()
    plt.savefig('outputs/'+country+'/data.png')
    # Log an artifact (output file)   
    return True

def draw_drifts(country, drifts, drift_data, train_data):
  #fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8), sharex=True)
  fig, ax1 = plt.subplots(figsize=(10,5))
  for d in range(len(drift_data)):
    ax1.plot(drift_data[d].fillna(method='bfill'), label=d)
  
  #print(drifts)
  #print(train_data.loc[drifts].values[0])
  #print(train_data.loc[drifts].values[0,:])
  #print(train_data.loc[drifts].values[:,0])
  ax1.bar(x=train_data.loc[drifts].index, height=train_data.loc[drifts].values[:,0], width=2, color='black')
  #ax1.annotate(train_data.loc[drifts].index, xy=(10, -100),  xycoords='axes points', xytext=(train_data.loc[drifts].index, -150), textcoords='data')
  """
  for k, drift_point in enumerate(pd.to_datetime(drifts, format="%m/%d/%y")):
      #print(drift_point)
      #print(drift_point.date())
      #ax1.annotate(k, xy=(10, 100),  xycoords='axes points', xytext=(drift_point, -10), textcoords='data')
      #ax1.annotate(drift_point.date().strftime('%Y-%m-%d'), xy=(10, -100),  xycoords='axes points', xytext=(drift_point-delta(days=10), -150), textcoords='data', rotation=90)
      #ax1.annotate(drift_point.date(), xy=(10, -100),  xycoords='axes points', xytext=(train_data.loc[drifts].index, -150), textcoords='data')
      #ax1.annotate(1, xy=(10, -100),  xycoords='data', xytext=(train_data.loc[drift_point],strftime('%d/%m/%y').index, -150))
      #ax1.annotate(k, xy=(10, -5000),  xycoords='axes points', xytext=(dt.strptime(drift_point, "%m/%d/%y")-timedelta(days=20), -5500), textcoords='data')
  """
  #ax2.plot(train_data.cumsum())
  plt.xticks([])
  plt.title(country)
  plt.legend()
  plt.savefig('outputs/'+country+'/drifts.png')
  # Log an artifact (output file)

  return True

def draw_diffs(country, diffs):
  #fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8), sharex=True)
  fig, ax1 = plt.subplots(figsize=(10,5))
  for d in range(len(diffs)):
    ax1.plot(diffs[d].fillna(method='bfill'), label=d)
  plt.xticks([])
  plt.title(country)
  plt.legend()
  plt.savefig('outputs/'+country+'/diffs.png')
  # Log an artifact (output file)

  return True

def draw_predictions(country, predictions, true, oracle=False):
    fig, ax = plt.subplots(figsize=(10,5))
    if oracle == False:
      ax.plot(predictions.reset_index(drop=True), label='pred')
      ax.plot(true.reset_index(drop=True), label='true')
      plt.xticks([])
      plt.title(country)
      plt.legend()
      plt.savefig('outputs/'+country+'/prediction.png')
    else:
      ax.plot(predictions.reset_index(drop=True), label='pred')
      ax.plot(true.reset_index(drop=True), label='oracle')
      plt.xticks([])
      plt.title('oracle')
      plt.legend()
      plt.savefig('outputs/'+country+'/oracle.png')
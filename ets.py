from statsmodels.tsa.exponential_smoothing.ets import ETSModel

import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


def train_ets(train_data):
    #ets_order = pmautoarima(train_data, max_p=7, d=1, max_q=7, m=7, seasonal=False, trace=True, information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
    #print(train_data.shape)
    #print(train_data.values.flatten())
    ets_base = ETSModel(train_data.values.flatten(), error='add', trend='add', seasonal='add', seasonal_periods=7)
    #log_param("order", arima_order.order)
    ets_model = ets_base.fit()
    print(ets_model.summary())
    return ets_model

def update_ets(model, test_data):
    ets_base = ETSModel(test_data.values.flatten(), error='add', trend='add', seasonal='add', seasonal_periods=7)
    model.initialize(ets_base, model.params)
    print(model.summary())
    return model

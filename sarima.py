from pmdarima.arima import auto_arima as pmautoarima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def train_arima(train_data, sarima=False):
    if sarima == True:
        # arima_order = pmautoarima(train_data, max_p=8, d=1, max_q=7, m=4, seasonal=True, trace=True, information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
        arima_order = pmautoarima(train_data, max_p=8, d=1, max_q=7, m=4, seasonal=True, trace=True, with_intercept=True,
                                  information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
        if (arima_order.with_intercept == True):
            arima_base = SARIMA(train_data, order=arima_order.order, seasonal_order=arima_order.seasonal_order, trend='ct')
        else:
            arima_base = SARIMA(train_data, order=arima_order.order, seasonal_order=arima_order.seasonal_order)
        #log_param("order", arima_order.order)
        #log_param("seasonal_order", arima_order.seasonal_order)
    else:
        arima_order = pmautoarima(train_data, max_p=8, d=1, max_q=7, m=4, seasonal=False, trace=True, information_criterion='aic', suppress_warnings=True, maxiter = 50, stepwise=True)
        arima_base = ARIMA(train_data, order=arima_order.order)
        #log_param("order", arima_order.order)
    arima_model = arima_base.fit()
    return arima_model

def update_arima(model, test_data):
    arima_updated = model.apply(test_data)
    return arima_updated

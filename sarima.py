import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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

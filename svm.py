import sklearn
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


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
    study.optimize(objective_svm, n_trials=20, show_progress_bar=True)
    params = study.best_params
    best_svr = SVR(kernel=params['kernel'], C=params['C'], epsilon=params['epsilon'], tol=params['tolerance'])
    best_svr.fit(x, y)
    return best_svr

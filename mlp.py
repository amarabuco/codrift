import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers

""" def objective_mlp(trial):
    clear_session()
    
    model = Sequential()
    model.add(layers.Dense(units= trial.suggest_categorical('units', [8, 16, 32, 64, 100, 200]), input_shape=(14, 1)))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=trial.suggest_categorical('activation', ['relu', 'linear', 'tanh']),) )
    #model.add(layers.Dense(1, activation='linear') )

    score = np.zeros(3)
    for i in range(3):
        # We compile our model with a sampled learning rate.
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

        model.fit(
            X_train_mm,
            y_train,
            validation_data=(X_val_mm, y_val),
            shuffle=False,
            batch_size=32,
            epochs=50,
            verbose=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        )

        # Evaluate the model accuracy on the validation set.
        score[i] = model.evaluate(X_val_mm, y_val, verbose=0)
    return -score.mean()   

def mlp(input_size, units, activation):
    model = Sequential()
    model.add(layers.Dense(units= units, input_shape=(input_size, 1)))
    #model.add(layers.Dense(1, activation='linear' ))
    model.add(layers.Dense(1, activation=activation ))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_mlp(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='mlp_study')
    study.optimize(objective_mlp, n_trials=20, show_progress_bar=True)
    params = study.best_params
    log_params(study.best_params)
    #best_lstm = lstm(input_size=14, units=params['units'], activation='linear') #variar o input-size (FIX)
    best_mlp = mlp(input_size=14, units=params['units'], activation=params['activation']) #variar o input-size (FIX)
    best_mlp.fit(x,
                y,
                validation_data=(X_val_mm, y_val),
                shuffle=False,
                batch_size=32,
                epochs=100,
                verbose=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]) """
    return best_mlp


def train_mlp_old(x, y):
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
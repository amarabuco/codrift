import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers

def objective_lstm(trial):
    model = Sequential()
    model.add(layers.LSTM(units= trial.suggest_categorical('units', [8, 16, 32, 64, 100, 200]), input_shape=(14, 1)))
    #model.add(layers.LSTM(100, input_shape=(14, 1), dropout=0.2, return_sequences=True))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.LSTM(100, dropout=0.2))
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

def lstm(input_size, units, activation):
    model = Sequential()
    model.add(layers.LSTM(units= units, input_shape=(input_size, 1)))
    #model.add(layers.Dense(1, activation='linear' ))
    model.add(layers.Dense(1, activation=activation ))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm(x, y):
    #optimization
    study = optuna.create_study(direction="maximize", study_name='lstm_study')
    study.optimize(objective_lstm, n_trials=20, show_progress_bar=True)
    params = study.best_params
    log_params(study.best_params)
    #best_lstm = lstm(input_size=14, units=params['units'], activation='linear') #variar o input-size (FIX)
    best_lstm = lstm(input_size=14, units=params['units'], activation=params['activation']) #variar o input-size (FIX)
    best_lstm.fit(x,
                y,
                validation_data=(X_val_mm, y_val),
                shuffle=False,
                batch_size=32,
                epochs=100,
                verbose=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    return best_lstm

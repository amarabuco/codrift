import torch
import torch.nn as nn
import torch.optim as optim

class ELM():
    def __init__(self, input_size, h_size, activation, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = 1
        self._device = device
        self.activation_name = activation

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = torch.zeros(self._h_size, device=self._device)

        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)

def objective_elm(trial):
    #h_size = trial.suggest_int("h_size", 2, 20)
    h_size = trial.suggest_categorical('h_size', [8, 16, 32, 64, 100, 200])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh", "relu"])
    # Generate the model.
    results = np.zeros(10)
    
    for i in range(10):
        model = ELM(input_size=14, h_size=h_size, activation=activation, device=device) #variar o input-size (FIX)
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
    best_elm = ELM(input_size=14, h_size=params['h_size'], activation=params['activation'], device=device) #variar o input-size (FIX)
    best_elm.fit(x, y)
    return best_elm
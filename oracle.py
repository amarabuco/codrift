import numpy as np
import pandas as pd

def get_oracle(predictions, true, order=1):
  df_error = predictions.iloc[1:].rsub(np.array(true.iloc[1:]), axis=0).abs()
  oracle = {}
  selection = []
  if order == 1:
    for row in df_error.rank(axis=1).idxmin(axis=1).items():
      oracle[row[0]] = predictions.at[row[0], row[1]]
      selection.append(row[1])
  else:
    for row in df_error.rank(axis=1).idxmax(axis=1).items():
      oracle[row[0]] = predictions.at[row[0], row[1]]
      selection.append(row[1])
  return pd.Series(selection).reset_index(drop=True), pd.Series(oracle).reset_index(drop=True)


def get_oracle(predictions, true):
  df_error = predictions.iloc[1:].rsub(np.array(true.iloc[1:]), axis=0).abs()
  oracle = {}
  selection = []
  for row in df_error.rank(axis=1).idxmin(axis=1).items():
    oracle[row[0]] = predictions.at[row[0], row[1]]
    selection.append(row[1])
  return selection, pd.Series(oracle)

def random_walk(data):
    return data.shift(1).dropna()
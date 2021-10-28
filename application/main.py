import pandas as pd


def get_from_csv(path):
    """ Return predictors and targets from a single csv """
    base = pd.read_csv(path)
    predictors = base.iloc[:, 0:11].values
    targets = base.iloc[:, 11].values
    return predictors, targets



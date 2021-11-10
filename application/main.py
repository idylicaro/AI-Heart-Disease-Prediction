import pandas as pd
import numpy as np
from sklearn import preprocessing

# 18 inputs in initial layer

def get_from_csv(path):
    """ Return predictors and targets from a single csv """
    base = pd.read_csv(path)
    # TODO: pre-process attributes
    predictors = base.iloc[:, 0:11].values
    targets = base.iloc[:, 11].values

    # escaler = preprocessing.MinMaxScaler()
    escaler = preprocessing.StandardScaler()
    escalerLabel = preprocessing.LabelBinarizer()

    age = escaler.fit_transform(np.array(predictors[:, 0]).reshape(-1, 1))
    escalerLabel.fit(['M', 'F'])
    sex = escalerLabel.transform(np.array(predictors[:, 1]).reshape(-1, 1))
    escalerLabel.fit(['ATA', 'NAP', 'ASY', 'TA'])
    chestPainType = escalerLabel.transform(np.array(predictors[:, 2]).reshape(-1, 1))
    restingBP = escaler.fit_transform(np.array(predictors[:, 3]).reshape(-1, 1))
    cholesterol = escaler.fit_transform(np.array(predictors[:, 4]).reshape(-1, 1))
    fastingBS = escaler.fit_transform(np.array(predictors[:, 5]).reshape(-1, 1))
    escalerLabel.fit(['Normal', 'ST', 'LVH'])
    restingECG = escalerLabel.transform(np.array(predictors[:, 6]).reshape(-1, 1))
    maxHR = escaler.fit_transform(np.array(predictors[:, 7]).reshape(-1, 1))
    escalerLabel.fit(['N', 'Y'])
    exerciseAngina = escalerLabel.fit_transform(np.array(predictors[:, 8]).reshape(-1, 1))
    oldpeak = escaler.fit_transform(np.array(predictors[:, 9]).reshape(-1, 1))
    escalerLabel.fit(['Up', 'Flat', 'Down'])
    sT_Slope = escalerLabel.transform(np.array(predictors[:, 10]).reshape(-1, 1))

    predictors = np.column_stack((age, sex, chestPainType, restingBP, cholesterol, fastingBS, restingECG, maxHR,
                                  exerciseAngina, oldpeak, sT_Slope))

    return predictors, targets


get_from_csv('heart.csv')
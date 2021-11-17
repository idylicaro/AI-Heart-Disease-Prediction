import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
import json


# 18 inputs in initial layer

def evaluate(json):
    classifier_mlp = MLPClassifier(hidden_layer_sizes=(10, 6, 3), max_iter=5000, activation='logistic', solver='adam',
                                   random_state=1, momentum=0.8, learning_rate_init=0.1, verbose=False)
    classifier_mlp.coefs_ = read_coef_file('weight.json')
    data = pd.read_csv('heart.csv')
    X, _ = get_from_csv(data)
    # TODO: HAS ERROR
    result = classifier_mlp.predict_proba(X[0].reshape(-1, 1))
    print('result:',result)


def get_from_csv(data):
    """ Return predictors and targets from a single csv """
    predictors = data.iloc[:, 0:11].values
    targets = data.iloc[:, 11].values

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


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_coef_file(coefs):
    data = {'coefs': coefs}
    pd.Series(coefs).to_json('weight.json', orient='split')
    # with open('weight.json', 'w') as outfile:
    #     json.dump(data, outfile, cls=NumpyEncoder)


def read_coef_file(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data['data']


def train_model(data_set_path, classifier):
    kf = KFold(n_splits=10)
    clf = MLPClassifier()
    data = pd.read_csv(data_set_path)

    # Splitting the dataset into  training and validation sets
    training_set, validation_set = train_test_split(data, test_size=0.2, random_state=21)

    X_train, Y_train = get_from_csv(training_set)
    X_val, Y_val = get_from_csv(validation_set)

    # Fitting the training data to the network
    classifier.fit(X_train, Y_train)

    # Predicting y for X_val
    Y_pred = classifier.predict(X_val)

    # Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(Y_pred, Y_val)

    # Printing the accuracy
    print(f"Accuracy of Classifier : {accuracy(cm)}")
    return classifier.coefs_


# quantidade_de_neuronios_na_camada_oculta = (numero de entradas + numero de saida) / 2
# Initializing the MLPClassifier
classifier_mlp = MLPClassifier(hidden_layer_sizes=(10, 6, 3), max_iter=5000, activation='logistic', solver='adam',
                               random_state=1, momentum=0.8, learning_rate_init=0.1, verbose=False)

print('\n==MLP==')
generate_coef_file(train_model("heart.csv", classifier_mlp))
# print(read_coef_file('weight.json'))
evaluate(None)

print('\n')

classifier_ppn = Perceptron(max_iter=5000, verbose=False, n_iter_no_change=10)

print('\n==Perceptron==')
train_model("heart.csv", classifier_ppn)
print('\n')

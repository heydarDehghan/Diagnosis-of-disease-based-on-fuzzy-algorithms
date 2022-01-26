import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Info:
    def __init__(self, alpha_cut, accuracy, cluster_number):
        ''' '''
        self.alpha_cut = alpha_cut
        self.accuracy = accuracy
        self.cluster_number = cluster_number


class Data:
    ''''''

    def __init__(self, data, data_range=(0, -1), label_range=-1, split_rate=0.2, bias=True, normal=False,
                 deleteNan=True):
        self.trainData: np.array
        self.testData: np.array
        self.trainLabel: np.array
        self.testLabel: np.array
        self.split_rare = split_rate
        self.data = data
        self.bias = bias
        self.data_range = data_range
        self.label_range = label_range
        self.deleteNan = deleteNan
        self.normal = normal
        self.class_list = None
        self.prepare_data()

    def prepare_data(self):

        temp_data = np.array(self.data[:, self.data_range[0]:self.data_range[1]], dtype=np.float64)
        temp_label = np.array(self.data[:, self.label_range], dtype=np.float64).reshape(temp_data.shape[0], 1)

        self.data = np.append(temp_data, temp_label, axis=1)

        self.data[:, self.label_range] = np.unique(self.data[:, self.label_range], return_inverse=True, )[1]
        if self.bias:
            self.data = np.insert(self.data, 0, 1, axis=1)

        if self.deleteNan:
            self.data = self.data[~np.isnan(self.data).any(axis=1), :]

        if self.normal:
            self.normalizer()



        self.trainData, self.testData, self.trainLabel, self.testLabel = train_test_split(
            self.data[:, :-1],
            self.data[:, -1],
            test_size=self.split_rare,
            random_state=42)

        self.class_list = np.unique(self.trainLabel)
        self.class_list.sort()

    def normalizer(self):
        scaler = MinMaxScaler()
        self.data[:, :-1] = scaler.fit_transform(self.data[:, :-1])


def load_data(path, array=True):
    train = pd.read_csv(path)
    if array:
        train = train.to_numpy()
    return train


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

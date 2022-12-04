import numpy as np


def get_accuracy(data, target):
    accuracy_list = []
    for x, y in zip(data, target):
        is_correct = (np.array(x) != -1) & (np.array(x) == np.array(y))
        is_data = np.array(y) != -1
        accuracy_list.append(len(is_correct[is_correct == True]) / len(is_data[is_data == True]))
    return accuracy_list

import torch
import numpy as np


def get_accuracy(data, target):
    accuracy_list = []
    for x, y in zip(data, target):
        x, y = x.cpu(), y.cpu()
        is_correct = (np.array(x) != -1) & (np.array(x) == np.array(y))
        is_data = np.array(y) != -1
        accuracy_list.append(len(is_correct[is_correct == True]) / len(is_data[is_data == True]))
    return accuracy_list


def get_recall(data, target, recall_list):
    for x, y in zip(data, target):
        x, y = x.cpu(), y.cpu()
        for index, label in enumerate(y):
            if label == -1:
                break
            if x[index] == label:
                recall_list[label.type(torch.int32)][0] += 1
            else:
                recall_list[label.type(torch.int32)][1] += 1
    return recall_list


def get_precision(data, target, precision_list):
    for x, y in zip(data, target):
        x, y = x.cpu(), y.cpu()
        for index, label in enumerate(x):
            if label == -1:
                break
            if y[index] == label:
                precision_list[label.type(torch.int32)][0] += 1
            else:
                precision_list[label.type(torch.int32)][1] += 1
    return precision_list


def calculate(data):
    answer = []
    for i in data:
        if i[0] != 0:
            answer.append(i[0] / (i[0] + i[1]))
    return np.mean(answer)

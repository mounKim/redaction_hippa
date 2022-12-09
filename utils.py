import os
import torch
import numpy as np
import xml.etree.ElementTree as elemTree

from label import *
from gensim.models import Word2Vec


def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def my_collate(samples):
    inputs = [sample['input'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    pad_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    pad_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    return pad_inputs, pad_labels


def token_normalization(text):
    text = text.lower()
    number = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for n in number:
        text = text.replace(n, '0')
    return text


def preprocess_i2b2_2014(path, rigid):
    texts = []
    labels = []
    for data in os.listdir(path):
        tree = elemTree.parse(os.path.join(path, data))
        sentence = tree.findall('TEXT')[0].text
        sentence = sentence.replace('\n', ' ')
        texts.append(token_normalization(sentence))
        label = []
        for tag in tree.findall('TAGS')[0]:
            tmp = tag.get('TYPE')
            if is_i2b2_2014(tmp, rigid):
                label.append(Label(tmp, tag.get('start'), tag.get('end')))
        labels.append(label)
    return texts, labels


def casing_features(tokens):
    features = []
    for token in tokens:
        if token.isupper():
            features.append(0)
        elif token.islower():
            features.append(1)
        else:
            features.append(2)
    return features


def make_label(token_pos, labels):
    pos = 0
    label_list = []
    for label in labels:
        while pos < len(token_pos) and token_pos[pos] < label.start:
            label_list.append(0)
            pos += 1
        while pos < len(token_pos) and label.start <= token_pos[pos] < label.end:
            if label.tag is None:
                label_list.append(0)
            else:
                label_list.append(label.tag.value)
            pos += 1
    label_list.extend([0] * (len(token_pos) - len(label_list)))
    return label_list


def spacing_features(sentence, tokenize_sentence):
    features = []
    start_pos = []
    pointer = 0
    for token in tokenize_sentence:
        if token == '#' or token == '[UNK]':
            features.append(0)
            start_pos.append(pointer)
            continue
        if token.startswith('##'):
            token = token[2:]
        size = len(token)
        feature = 0
        while True:
            if sentence[pointer:pointer + size] == token:
                features.append(feature)
                start_pos.append(pointer)
                pointer += size
                break
            pointer += 1
            feature += 1
    return features, start_pos


def tokenize_alldata(dataset, tokenizer):
    tokens = []
    for data in dataset:
        tokens.append(tokenizer.tokenize(data))
    return tokens


def make_word2vec(sentences, min_count):
    model = Word2Vec(sentences=sentences, vector_size=200, min_count=min_count)
    model.save('word2vec.model')
    return model

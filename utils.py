import os
import torch
import xml.etree.ElementTree as elemTree

from label import *
from gensim.models import Word2Vec


def preprocess_i2b2_2014(path):
    texts = []
    labels = []
    for data in os.listdir(path):
        tree = elemTree.parse(os.path.join(path, data))
        texts.append(tree.findall('TEXT')[0].text)
        label = []
        for tag in tree.findall('TAGS')[0]:
            tmp = tag.get('TYPE')
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
    return torch.Tensor(features)


def spacing_features(sentence, tokenize_sentence):
    features = [0]
    sentence = sentence.replace('\n', ' ')
    pointer = len(tokenize_sentence[0])
    for token in tokenize_sentence[1:]:
        if token.startswith('##'):
            token = token[2:]
        size = len(token)
        feature = 0
        while True:
            if sentence[pointer:pointer + size] == token:
                pointer += size
                features.append(feature)
                break
            pointer += 1
            feature += 1
    return torch.Tensor(features)


def tokenize_alldata(dataset, tokenizer):
    tokens = []
    for data in dataset:
        tokens.append(tokenizer.tokenize(data))
    return tokens


def make_word2vec(sentences, min_count):
    model = Word2Vec(sentences=sentences, vector_size=200, min_count=min_count)
    model.save('word2vec.model')
    return model

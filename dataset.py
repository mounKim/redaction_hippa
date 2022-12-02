import numpy as np
import torch.utils.data as data
from utils import *


class ClinicalDataset(data.Dataset):
    def __init__(self, data_list, label_list, tokenizer, word2vec_model):
        super(ClinicalDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model

    def __getitem__(self, index):
        sentence = self.data_list[index]
        label = self.label_list[index]
        tokenize_sentence = self.tokenizer.tokenize(sentence)
        spacing_feature, label_feature = spacing_features(sentence, tokenize_sentence)
        word2vec_sentence = []
        for token in tokenize_sentence:
            if token in self.word2vec_model.wv.key_to_index.keys():
                word2vec_sentence.append(self.word2vec_model.wv[token].copy())
            else:
                word2vec_sentence.append([0.] * 200)
        # character birnn (25)
        return {'input': torch.cat([torch.tensor(np.array(word2vec_sentence), dtype=torch.float32),
                                    torch.tensor(np.array(casing_features(tokenize_sentence))).unsqueeze(1),
                                    torch.tensor(np.array(spacing_feature)).unsqueeze(1)
                                    ], 1), 'label': torch.Tensor(make_label(label_feature, label))}

    def __len__(self):
        return len(self.data_list)

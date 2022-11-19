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
        tokenize_sentence = self.tokenizer.tokenize(sentence)
        word2vec_sentence = []
        for token in tokenize_sentence:
            if token in self.word2vec_model.wv.key_to_index.keys():
                word2vec_sentence.append(torch.Tensor(self.word2vec_model.wv[token].copy()))
            else:
                word2vec_sentence.append(torch.Tensor([0.] * 250))
        # character birnn (25)
        return torch.cat([word2vec_sentence, casing_features(tokenize_sentence),
                          spacing_features(sentence, tokenize_sentence)], 1), self.label_list[index]

    def __len__(self):
        return len(self.data_list)

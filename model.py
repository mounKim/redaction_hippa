import torchcrf
import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self, training=True):
        super(OurModel, self).__init__()
        self.token_birnn = nn.RNN(227, 100, bidirectional=True)
        self.tag_predict = nn.Linear(200, 19)
        self.crf = torchcrf.CRF(19)
        self.training = training

    def forward(self, x):
        x = self.token_birnn(x)
        x = self.tag_predict(x)
        if self.training:
            x = self.crf(x)
        else:
            x = self.crf.decode(x)
        return x

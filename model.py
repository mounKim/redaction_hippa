import torch
import torchcrf
import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.token_birnn = nn.RNN(202, 100, bidirectional=True)
        self.tag_predict = nn.Linear(200, 19)
        self.softmax = nn.Softmax(dim=2)
        self.crf = torchcrf.CRF(19, batch_first=True)

    def forward(self, x):
        x = self.token_birnn(x)
        x = self.tag_predict(x[0])
        x = self.softmax(x)
        x = self.crf.decode(x)

        return torch.Tensor(x)

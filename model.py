import torch
import torchcrf
import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self, training=True, reduction='sum'):
        super(OurModel, self).__init__()
        self.token_birnn = nn.RNN(202, 100, bidirectional=True)
        self.tag_predict = nn.Linear(200, 19)
        self.softmax = nn.Softmax(dim=2)
        self.crf = torchcrf.CRF(19, batch_first=True)
        self.training = training
        self.reduction = reduction

    def forward(self, x, y):
        mask = torch.where(y == -1, 0, 1)
        x = self.token_birnn(x)
        x = self.tag_predict(x[0])
        x = self.softmax(x)
        if self.training:
            x = -1 * self.crf(x, y.type(torch.int64), mask=mask.type(torch.uint8), reduction=self.reduction)
        else:
            x = self.crf.decode(x)
        return x

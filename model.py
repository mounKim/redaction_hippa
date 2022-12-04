import torch
import torchcrf
import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self, reduction='sum'):
        super(OurModel, self).__init__()
        self.token_birnn = nn.RNN(202, 100, bidirectional=True)
        self.tag_predict = nn.Linear(200, 19)
        self.softmax = nn.Softmax(dim=2)
        self.crf = torchcrf.CRF(19, batch_first=True)
        self.reduction = reduction

    def forward(self, x, y):
        mask = torch.where(y == -1, 0, 1).type(torch.uint8)
        x = self.token_birnn(x)
        x = self.tag_predict(x[0])
        x = self.softmax(x)
        loss = -1 * self.crf(x, y.type(torch.int64), mask=mask, reduction=self.reduction)
        label = self.crf.decode(x, mask=mask)
        label = [torch.tensor(x) for x in label]
        label = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=-1)
        return loss, label

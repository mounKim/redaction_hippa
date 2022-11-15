import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.token_birnn = nn.RNN(227, 100, bidirectional=True)

    def forward(self, x):
        x = self.token_birnn(x)
        return x

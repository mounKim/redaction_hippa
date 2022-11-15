import torch
import torch.nn as nn

from tqdm import tqdm


def train(model, args, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    model.train()

    print('start training')
    for epoch in range(args.epochs):
        train_loss = []
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()


def predict(model, args, dataloader):
    model.eval()
    predict_labels = []

    print('start predicting')
    for step, batch in tqdm(enumerate(dataloader)):
        pass

    return predict_labels

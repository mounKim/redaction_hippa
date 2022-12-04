import torch
import numpy as np

from metric import *
from tqdm import tqdm


def train(model, args, dataloader):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    model.train()

    print('start training')
    for epoch in range(args.epochs):
        train_loss = []
        predict_label = []
        real_label = []
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_input, b_label = batch
            loss, label = model(b_input, b_label)
            loss.backward()
            train_loss.append(loss.item())
            predict_label.append(label)
            real_label.append(b_label)
            optimizer.step()

        predict_label = torch.cat(predict_label).tolist()
        real_label = torch.cat(real_label).tolist()
        avg_train_loss = np.mean(train_loss)
        avg_train_acc = get_accuracy(predict_label, real_label)
        print("Epoch {0},  Average training loss: {1:.2f},  Average training accuracy: {2:.4f}"
              .format(epoch, avg_train_loss, avg_train_acc))


def predict(model, args, dataloader):
    model.eval()
    predict_labels = []

    print('start predicting')
    for step, batch in tqdm(enumerate(dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            out = model(batch)
        predict_labels.append(out)

    predict_labels = torch.cat(predict_labels).tolist()
    return predict_labels

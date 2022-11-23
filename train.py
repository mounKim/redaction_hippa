import torch
import numpy as np

from tqdm import tqdm


def train(model, args, dataloader):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    model.train()

    print('start training')
    for epoch in range(args.epochs):
        train_loss = []
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_inputs, b_labels = batch
            loss = model(b_inputs, b_labels)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

        avg_train_loss = np.mean(train_loss)
        print("Epoch {0},  Average training loss: {1:.2f}".format(epoch, avg_train_loss))


def predict(model, args, dataloader):
    model.eval()
    predict_labels = []

    print('start predicting')
    for step, batch in tqdm(enumerate(dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            out = model(batch, training=False)
        predict_labels.append(out)

    predict_labels = torch.cat(predict_labels).tolist()
    return predict_labels

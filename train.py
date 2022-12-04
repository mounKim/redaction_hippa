import torch
from metric import *
from tqdm import tqdm


def train(model, args, dataloader):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    model.train()

    print('start training')
    for epoch in range(args.epochs):
        train_loss = []
        train_acc = []
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_input, b_label = batch
            loss, label = model(b_input, b_label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.extend(get_accuracy(label, b_label))

        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc)
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

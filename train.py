from metric import *


def train(model, args, dataloader):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    model.train()

    print('start training')
    for epoch in range(args.epochs):
        train_loss = []
        train_acc = []
        train_rec = [[0, 0] for _ in range(19)]
        train_pre = [[0, 0] for _ in range(19)]
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_input, b_label = batch
            loss, label = model(b_input, b_label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.extend(get_accuracy(label, b_label))
            train_rec = get_recall(label, b_label, train_rec)
            train_pre = get_precision(label, b_label, train_pre)

        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc)
        avg_train_rec = calculate(train_rec)
        avg_train_pre = calculate(train_pre)
        print("Epoch {0},  Training loss: {1:.2f},  Training accuracy: {2:.4f},  Training recall: {3:.4f},  "
              "Training precision: {4:.4f}"
              .format(epoch, avg_train_loss, avg_train_acc, avg_train_rec, avg_train_pre))


def predict(model, args, dataloader):
    model.eval()
    predict_labels = []

    print('start predicting')
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            out = model(batch)
        predict_labels.append(out)

    predict_labels = torch.cat(predict_labels).tolist()
    return predict_labels

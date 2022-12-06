from metric import *


def train(model, args, train_dataloader, val_dataloader):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_acc = []
        if args.rigid_labeling:
            train_rec = [[0, 0] for _ in range(19)]
            train_pre = [[0, 0] for _ in range(19)]
        else:
            train_rec = [[0, 0] for _ in range(4)]
            train_pre = [[0, 0] for _ in range(4)]
        for step, batch in enumerate(train_dataloader):
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

        avg_val_acc, avg_val_rec, avg_val_pre = predict(model, args, val_dataloader)

        print("Epoch {0},  Training loss: {1:.2f},  Training accuracy: {2:.4f},  Validation accuracy: {3:.4f},  "
              "Validation recall: {4:.4f},  Validation precision: {5:.4f}"
              .format(epoch, avg_train_loss, avg_train_acc, avg_val_acc, avg_val_rec, avg_val_pre))


def predict(model, args, dataloader):
    model.eval()
    val_acc = []
    if args.rigid_labeling:
        val_rec = [[0, 0] for _ in range(19)]
        val_pre = [[0, 0] for _ in range(19)]
    else:
        val_rec = [[0, 0] for _ in range(4)]
        val_pre = [[0, 0] for _ in range(4)]

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        b_input, b_label = batch
        with torch.no_grad():
            loss, label = model(b_input, b_label)
        val_acc.extend(get_accuracy(label, b_label))
        val_rec = get_recall(label, b_label, val_rec)
        val_pre = get_precision(label, b_label, val_pre)

    return np.mean(val_acc), calculate(val_rec), calculate(val_pre)

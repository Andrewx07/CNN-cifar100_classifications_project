import torch
import numpy as np


def training_cycle(model, training_time, optimizer, criterion, dataset):
    graphic1 = []
    graphic2 = []
    running_epoch = []
    accuracy = []
    acc_val = []

    for epoch in range(training_time):
        train_loss = 0.0
        valid_loss = 0.0

        for inputs, labels in dataset["train_dataloader"]:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        for inputs, labels in dataset["validation_dataloader"]:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

        running_epoch.append(float(epoch))

        accuracy, class_correct_test, class_total_test, test_loss = testing_func(
            dataset, model, criterion
        )

        train_loss = train_loss / len(dataset["train_dataloader"])
        print("\n[Para la epoca", epoch + 1, "] \nloss:", train_loss)
        graphic1.append(train_loss)

        valid_loss = valid_loss / len(dataset["validation_dataloader"])
        print("\nval_loss:", valid_loss)
        graphic2.append(valid_loss)

        print(f"\nTest Loss: {test_loss}")

        print(
            f"\nTest Accuracy : {float(100. * np.sum(class_correct_test) / np.sum(class_total_test))}% where {int(np.sum(class_correct_test))} of {int(np.sum(class_total_test))} were pmodelicted correctly"
        )

    return (
        graphic1,
        graphic2,
        running_epoch,
        accuracy,
        class_total_test,
        class_correct_test,
        test_loss,
    )


def testing_func(dataset, model, criterion):
    class_correct_test = list(0.0 for i in range(100))
    class_total_test = list(0.0 for i in range(100))
    test_loss = 0.0
    accuracy = []
    for images, lab in dataset["test_dataloader"]:
        if torch.cuda.is_available():
            images, lab = images.cuda(), lab.cuda()
        output = model(images)
        loss = criterion(output, lab)
        test_loss += loss.item()
        _, pmodel = torch.max(output, 1)
        correct = np.squeeze(pmodel.eq(lab.data.view_as(pmodel)))

        for idx in range(dataset["batch_size"]):
            label = lab[idx]
            class_correct_test[label] += correct[idx].item()
            class_total_test[label] += 1

        test_loss = test_loss / len(dataset["test_dataloader"])

        accuracy.append(float(np.sum(class_correct_test) / np.sum(class_total_test)))
    return accuracy, class_correct_test, class_total_test, test_loss

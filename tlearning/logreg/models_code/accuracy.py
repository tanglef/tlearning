import torch


def get_accuracy_torch(logreg_cifar, test_loader):
    total, correct = 0, 0
    for images, labels in test_loader:
        images = images.view(-1, 3*32*32).requires_grad_()
        # Forward
        outputs = logreg_cifar(images)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = correct / total
    return accuracy

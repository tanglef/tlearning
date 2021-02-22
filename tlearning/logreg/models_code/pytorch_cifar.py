"""
Logistic regression with CIFAR10 dataset

sources:
- https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/  # noqa
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, "data")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

input_size = 3 * 32 * 32
num_classes = 10
batch_size = 100
n_iters = 1e4
num_epochs = n_iters / 50000 * batch_size
num_epochs = int(num_epochs)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

####################
# Model with Pytorch
####################


class Logreg(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Logreg, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        output = self.linear(x)
        return output


###################
# CIFAR 10: 60k images of 32x32 with 10 class
###################

if __name__ == '__main__':

    ###################
    # Train model
    ###################
    get_accuracy = True
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # nll_loss(log_softmax(x, dim=-1), target)
    logreg_cifar = Logreg(input_size, num_classes)
    loss_f = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    sgd = torch.optim.SGD(logreg_cifar.parameters(),
                          lr=learning_rate, momentum=0)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images and zero out the optimizer
            images = images.view(-1, input_size).requires_grad_()
            sgd.zero_grad()

            # Forward
            outputs = logreg_cifar(images)

            # Backward
            loss = loss_f(outputs, labels)
            loss.backward()

            # Step
            sgd.step()

    # Saves only parameters
    torch.save({'epoch': num_epochs,
                'model_state_dict': logreg_cifar.state_dict(),
                'optimizer_state_dict': sgd.state_dict(),
                'loss': loss
                }, os.path.join(data_dir, "tch_model.pth"))

    if get_accuracy:
        total, correct = 0, 0
        for images, labels in test_loader:
            images = images.view(-1, input_size).requires_grad_()
            # Forward
            outputs = logreg_cifar(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = correct / total
        print(accuracy)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Adam
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms

from PIL import Image

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Create a simple CNN

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3,
                              out_channels=out_channels,
                              stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        # Computes the average of all activations in each channel
        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,
                                 self.unit4, self.unit5, self.unit6, self.unit7,
                                 self.pool2, self.unit8, self.unit9, self.unit10, self.unit11,
                                 self.pool3, self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output

# class MyNet():
#     def __init__(self):
#         # Transformation for image
#         self.transform = transforms.Compose([transforms.Resize(32),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize(
#                                                 mean=[0.485, 0.456, 0.406],
#                                                 std=[0.229, 0.224, 0.225])])

#         # Load our dataset
#         # TODO change to run from top folder
#         train_dataset = datasets.ImageFolder(
#             root='../dataset_generation/images/classification_clean/training_set',
#             transform=self.transform)

#         test_dataset = datasets.ImageFolder(
#             root='../dataset_generation/images/classification_clean/testing_set',
#             transform=self.transform)

#         # batch_size = 32

#         self.train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                         batch_size=4, shuffle=True,
#                                                         num_workers=4)

#         self.test_loader = torch.utils.data.DataLoader(test_dataset,
#                                                        batch_size=4, shuffle=True,
#                                                        num_workers=4)

#         self.training_set_size = len(train_dataset)
#         self.test_set_size = len(test_dataset)

#         print('There are {} images in the training set'.format(self.training_set_size))
#         print('There are {} images in the test set'.format(self.test_set_size))
#         print('There are {} batches in the train loader'.format(len(self.train_loader)))
#         print('There are {} batches in the test loader'.format(len(self.test_loader)))

#         # Check if gpu support is available
#         self.cuda_avail = torch.cuda.is_available()

#         # Create model, optimizer, and loss function
#         self.model = SimpleNet(num_classes=5)
#         print('created a model')

#         if self.cuda_avail:
#             self.model.cuda()

#         self.initial_lr = 0.001

#         # Define the optimizer and loss function
#         self.optimizer = Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=0.0001)
#         self.loss_fn = nn.CrossEntropyLoss()

#     def adjust_learning_rate(self, epoch):
#         """
#         Learning rate adjustment function that divides the learning rate by 
#         10 every 30 epochs
#         """
#         lr = self.initial_lr
#         adj = epoch / 30
#         lr = lr / (10**adj)

#         for param_group in self.optimizer.param_groups:
#             param_group["lr"] = lr

#     def save_models(self, epoch):
#         """
#         Save and evaluate the model.
#         """
#         torch.save(self.model.state_dict(), "mug_numeration_classifier.model")
#         print("checkpoint saved")

#     def test(self):
#         """
#         Returns accuracy of model determined from test images
#         """

#         self.model.eval()
#         test_acc = 0.0

#         # Iterate over the test loader
#         for i, (images, labels) in enumerate(self.test_loader):

#             if self.cuda_avail:
#                 images = Variable(images.cuda())
#                 labels = Variable(labels.cuda())

#             # Predict classes using images from the test set
#             outputs = self.model(images)

#             # Pick max prediction
#             _, prediction = torch.max(outputs.data, 1)

#             prediction_np = prediction.cpu().numpy()
#             labels_np = labels.data.cpu().numpy()
#             images_np = images.data.cpu().numpy()

#             # Compare to actual class to obtain accuracy
#             test_acc += torch.sum(prediction == labels.data).float()

#         # Compute the average acc and loss over all test images
#         test_acc = test_acc / self.test_set_size

#         return test_acc

#     def train(self, num_epochs):
#         """
#         Trains model with with num_epochs epochs.
#         """

#         print('training with {} epochs'.format(num_epochs))

#         best_acc = 0.0

#         for epoch in range(num_epochs):
#             self.model.train()
#             train_acc = 0.0
#             train_loss = 0.0

#             for i, (images, labels) in enumerate(self.train_loader):
#                 # Move images and labels to gpu if available
#                 if self.cuda_avail:
#                     images = Variable(images.cuda())
#                     labels = Variable(labels.cuda())
                
#                 # Clear all accumulated gradients
#                 self.optimizer.zero_grad()

#                 # Predict classes using images from the test set
#                 outputs = self.model(images)

#                 # Compute the loss based on the predictions and actual labels
#                 loss = self.loss_fn(outputs, labels)

#                 # Backpropagate the loss
#                 loss.backward()

#                 # Adjust parameters according to the computed gradients
#                 self.optimizer.step()

#                 train_loss += loss.cpu().item() * images.size(0)
#                 _, prediction = torch.max(outputs.data, 1)

#                 train_acc += torch.sum(prediction == labels.data).float()

#             # Call the learning rate adjustment function
#             self.adjust_learning_rate(epoch)

#             # Compute the average acc and loss over all 50,000 training images
#             train_acc = train_acc / self.training_set_size
#             train_loss = train_loss / self.training_set_size

#             # Evaluate on the test set
#             test_acc = self.test()

#             # Save the model if the test acc is greater than our current best
#             if test_acc > best_acc:
#                 self.save_models(epoch)
#                 best_acc = test_acc
#                 print("New best acc is {}, epoch {}".format(best_acc, epoch))

#             # Print the metrics
#             print("Epoch {}, Train Accuracy: {}, Train Loss: {}, Test Accuracy: {}".format(
#                 epoch, train_acc, train_loss, test_acc))

# def main():
#     net = MyNet()
#     net.train(num_epochs=200)

# if __name__ == "__main__":
#     main()

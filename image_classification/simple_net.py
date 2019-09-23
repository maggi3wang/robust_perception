import torch
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.datasets as datasets

from torch.optim import Adam
from torch.autograd import Variable

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
                                 self.unit4, self.unit5, self.unit6,self.unit7, 
                                 self.pool2, self.unit8, self.unit9, self.unit10, self.unit11,
                                 self.pool3, self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output

# Transformation for image
max_edge_length = 369   # TODO find this programatically
transform_ori = transforms.Compose([transforms.CenterCrop((max_edge_length)),
                                    transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
                                    
# Load our dataset
train_dataset = datasets.ImageFolder(root = '../dataset_generation/images/classification_clean/training_set',
                                     transform = transform_ori)

test_dataset = datasets.ImageFolder(root = '../dataset_generation/images/classification_clean/testing_set',
                                    transform = transform_ori)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=4, shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=4, shuffle=True,
                                          num_workers=4)


training_set_size = len(train_dataset)
test_set_size = len(test_dataset)

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer, and loss function
model = SimpleNet(num_classes=5)
print('created a model')

if cuda_avail:
    model.cuda()

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()
print('defined the optimizer and loss fn')

# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Save and evaluate the model
def save_models(epoch):
    torch.save(model.state_dict(), "mug_numeration_{}.model".format(epoch))
    print("checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0

    # Iterate over the test loader
    for i, (images, labels) in enumerate(test_loader):
        
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        
        # Pick max prediction
        _, prediction = torch.max(outputs.data, 1)

        prediction_np = prediction.cpu().numpy()
        labels_np = labels.data.cpu().numpy()

        images_np = images.data.cpu().numpy()

        for j, prediction in enumerate(prediction_np):
            if prediction != labels_np[j]:
                print('prediction: {}, actual: {}'.format(prediction, labels_np[j]))
                img = np.array(images_np[j] * 255, np.int32)
                fig = plt.figure(figsize=(32, 32))
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.show()

        # Compare to actual class to obtain accuracy
        test_acc += torch.sum(prediction == labels.data).float()

    # Compute the average ac and loss over all test images
    test_acc = test_acc / test_set_size

    return test_acc

print('There are {} images in the training set'.format(training_set_size))
print('There are {} images in the test set'.format(test_set_size))
print('There are {} batches in the train loader'.format(len(train_loader)))
print('There are {} batches in the test loader'.format(len(test_loader)))

def train(num_epochs):
    print('training with {} epochs'.format(num_epochs))

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            
            # Clear all accumulated gradients
            optimizer.zero_grad()

            # Predict classes using images from the test set
            outputs = model(images)

            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data).float()

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)    

        # Compute the average acc and loss over all 50,000 training images
        train_acc = train_acc / training_set_size
        train_loss = train_loss / training_set_size

        # Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {}, Train Loss: {}, Test Accuracy: {}, Best Accuracy: {}".format(
            epoch, train_acc, train_loss, test_acc, best_acc))

if __name__ == "__main__":
    train(1)

path = "mug_numeration_classifier.pth"
print('saving model to {}'.format(path))
torch.save(model.state_dict(), path)

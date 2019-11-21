import pycuda.driver as cuda
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

import csv
import errno
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

from ..image_classification.simple_net import SimpleNet


class MyNet():
    def __init__(self, model_file_number, models_dir, training_set_dir, test_set_dir, counterexample_set_dir):
        self.model_file_number = model_file_number

        # Transformation for image
        # TODO try with different sizes
        training_transform = transforms.Compose([
            transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        plain_transform = transforms.Compose([
            transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Load our dataset
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.package_dir, models_dir)

        # training_set_dir = '../data/experiment1/training_set' 
        training_set_dir = os.path.join(self.package_dir, training_set_dir)

        # test_set_dir = '../data/experiment1/test_set'
        test_set_dir = os.path.join(self.package_dir, test_set_dir)

        # counterexample_set_dir = '../data/experiment1/counterexample_set'
        counterexample_set_dir = os.path.join(self.package_dir, counterexample_set_dir)

        train_dataset = datasets.ImageFolder(root=training_set_dir, transform=training_transform)
        test_dataset = datasets.ImageFolder(root=test_set_dir, transform=plain_transform)
        counterexample_dataset = datasets.ImageFolder(root=counterexample_set_dir, transform=plain_transform)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=True, num_workers=0)

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=4, shuffle=True, num_workers=0)

        self.counterexample_loader = torch.utils.data.DataLoader(
            counterexample_dataset, batch_size=4, shuffle=True, num_workers=0)

        self.training_set_size = len(train_dataset)
        self.test_set_size = len(test_dataset)
        self.counterexample_set_size = len(counterexample_dataset)

        print('There are {} images in the training set'.format(self.training_set_size))
        print('There are {} images in the test set'.format(self.test_set_size))
        print('There are {} images in the counterexample set'.format(self.counterexample_set_size))

        print('There are {} batches in the train loader'.format(len(self.train_loader)))
        print('There are {} batches in the test loader'.format(len(self.test_loader)))
        print('There are {} batches in the counterexample loader'.format(len(self.counterexample_loader)))

        # Check if gpu support is available
        self.cuda_avail = torch.cuda.is_available()

        # Create model, optimizer, and loss function
        num_mugs = 5
        self.model = SimpleNet(num_classes=num_mugs)
        print('created a model')

        if self.cuda_avail:
            cuda.init()
            torch.cuda.set_device(0)
            print(cuda.Device(torch.cuda.current_device()).name())
            self.model.cuda()

        self.initial_lr = 0.001

        # Define the optimizer and loss function
        self.optimizer = Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()

        self.model_file_names = []

        self.train_accuracies = []
        self.test_accuracies = []
        self.counterexample_accuracies = []

        # torch.set_num_threads(1)

    def adjust_learning_rate(self, epoch):
        """
        Learning rate adjustment function that divides the learning rate by 
        10 every 30 epochs
        """
        lr = self.initial_lr
        adj = epoch / 30
        lr = lr / (10**adj)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def load_checkpoint(filename):
        start_epoch = 0
        model = SimpleNet(num_classes=5)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Move model to GPU
            model.cuda()

            # Move optimizer to GPU
            for state in optimizer.state.values():
                for k,v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        return model, optimizer, start_epoch

    def load_and_set_checkpoint(self, filename):
        self.model, self.optimizer, start_epoch = self.load_checkpoint(filename)

    def save_checkpoint(self, epoch):
        """
        Save and evaluate the model.
        """

        model_file_base = "mug_numeration_classifier_{:03d}_epoch_{:03d}.pth.tar".format(
            self.model_file_number, epoch)
        filename = os.path.join(self.models_dir, model_file_base)

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

        self.model_file_names.append(filename)

    def evaluate_accuracy(self, loader, set_size):
        """
        Returns accuracy of model determined from test images
        """

        self.model.eval()
        test_acc = 0.0

        # Iterate over the test loader
        for i, (images, labels) in enumerate(loader):

            if self.cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Predict classes using images from the test set
            outputs = self.model(images)

            # Pick max prediction
            _, prediction = torch.max(outputs.data, 1)

            prediction_np = prediction.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            images_np = images.data.cpu().numpy()

            # Compare to actual class to obtain accuracy
            test_acc += torch.sum(prediction == labels.data).float()

        # Compute the average acc and loss over all test images
        test_acc = test_acc / set_size

        return test_acc

    def train(self, num_epochs):
        """
        Trains model with num_epochs epochs.
        """

        print('training with {} epochs'.format(num_epochs))

        model_accuracies_csv = os.path.join(self.models_dir, 'model_{:03d}.csv'.format(self.model_file_number))
        f = open(model_accuracies_csv, 'w')
        f.write('epoch, training_loss, training_acc, test_acc, counterexample_acc, is_new_best,\n')
        f.flush()

        best_acc = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0

            for i, (images, labels) in enumerate(self.train_loader):
                # Move images and labels to gpu if available
                if self.cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                
                # Clear all accumulated gradients
                self.optimizer.zero_grad()

                # Predict classes using images from the test set
                outputs = self.model(images)

                # Compute the loss based on the predictions and actual labels
                loss = self.loss_fn(outputs, labels)

                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                self.optimizer.step()

                train_loss += loss.cpu().item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_acc += torch.sum(prediction == labels.data).float()

            # Call the learning rate adjustment function
            self.adjust_learning_rate(epoch)

            # Compute the average acc and loss over all training images
            train_acc = train_acc / self.training_set_size
            train_loss = train_loss / self.training_set_size

            # Evaluate on the test set
            test_acc = self.evaluate_accuracy(self.test_loader, self.test_set_size)
            counterexample_acc = self.evaluate_accuracy(self.counterexample_loader, self.counterexample_set_size)

            # Print the metrics
            # print("Epoch {}, Train Accuracy: {}, Train Loss: {}, Test Accuracy: {},"
            #     "Counterexample Accuracy: {}".format(
            #     epoch, train_acc, train_loss, test_acc, counterexample_acc))

            # self.train_accuracies.append(train_acc)
            # self.test_accuracies.append(test_acc)
            # self.counterexample_accuracies.append(counterexample_acc)

            f.write('{:3d}, {:2.5f}, {:1.5f}, {:1.5f}, {:1.5f}, {:1d},\n'.format(
                epoch, train_loss, train_acc, test_acc, counterexample_acc, test_acc > best_acc))
            f.flush()

            # Save the model if the test acc is greater than our current best
            if test_acc > best_acc:
                self.save_checkpoint(epoch)
                best_acc = test_acc
                print("New best acc is {}, epoch {}".format(best_acc, epoch))

        # Move last epoch to current model and delete all other models
        model_file_base = 'mug_numeration_classifier_{:03d}.pth.tar'.format(self.model_file_number)
        model_file_name = os.path.join(self.models_dir, model_file_base)

        print('copying {} to {}'.format(self.model_file_names[-1], model_file_name))
        shutil.copy(self.model_file_names[-1], model_file_name)

        for model_file_name in self.model_file_names:
            os.remove(model_file_name)

        # print('train_accuracies: {}'.format(self.train_accuracies))
        # print('test_accuracies: {}'.format(self.test_accuracies))
        # print('counterexample_accuracies: {}'.format(self.counterexample_accuracies))

        f.close()
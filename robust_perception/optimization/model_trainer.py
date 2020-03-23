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

import itertools

from PIL import Image

import csv
import errno
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import time
import random

from ..image_classification.simple_net import SimpleNet


class MyNet():
    def __init__(self, model_prefix, model_trial_number, num_data_added,
            models_dir, training_set_dirs, test_set_dir, counterexample_set_dir=None, num_workers=0):
        self.print = False

        self.model_prefix = model_prefix    # this is 'random' or 'counterex'
        self.model_trial_number = model_trial_number    # this is the trial number
        self.num_data_added = num_data_added    # this is the number of data added to initial dataset

        self.model_file_base_name = "{}_{:02d}_{:04d}".format(
            self.model_prefix, self.model_trial_number, self.num_data_added)

        # Transformation for image
        training_transform = transforms.Compose([
            transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        plain_transform = transforms.Compose([
            transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Load our dataset
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.package_dir, models_dir)

        training_datasets = []
        self.training_set_size = 0
        for training_set_dir in training_set_dirs:
            training_set_dir = os.path.join(self.package_dir, training_set_dir)
            training_set = datasets.ImageFolder(root=training_set_dir, transform=training_transform)
            training_datasets.append(training_set)

            self.training_set_size += len(training_set)
            print('There are {} images in this training set'.format(len(training_set)))

        test_set_dir = os.path.join(self.package_dir, test_set_dir)

        self.using_counterexample_set = False
        if counterexample_set_dir:
            self.using_counterexample_set = True

        if self.using_counterexample_set:
            counterexample_set_dir = os.path.join(self.package_dir, counterexample_set_dir)

        test_dataset = datasets.ImageFolder(root=test_set_dir, transform=plain_transform)

        self.batch_size = 32

        if self.print:
            print('training_datasets', training_datasets)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(training_datasets), batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        if self.using_counterexample_set:
            counterexample_dataset = datasets.ImageFolder(root=counterexample_set_dir, transform=plain_transform)
            self.counterexample_loader = torch.utils.data.DataLoader(
                counterexample_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            self.counterexample_set_size = len(counterexample_dataset)

        self.test_set_size = len(test_dataset)

        print('There are {} images in the test set'.format(self.test_set_size))

        if self.using_counterexample_set:
            print('There are {} images in the counterexample set'.format(self.counterexample_set_size))

        print('There are {} batches in the train loader'.format(len(self.train_loader)))
        print('There are {} batches in the test loader'.format(len(self.test_loader)))

        if self.using_counterexample_set:
            print('There are {} batches in the counterexample loader'.format(len(self.counterexample_loader)))

        # Check if gpu support is available
        self.cuda_avail = torch.cuda.is_available()

        # Create model, optimizer, and loss function
        self.num_classes = 5
        self.model = SimpleNet(num_classes=self.num_classes)

        if self.cuda_avail:
            cuda.init()
            torch.cuda.set_device(0)
            if self.print:
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
    def load_checkpoint(filename, use_gpu=True):
        start_epoch = 0
        model = SimpleNet(num_classes=5)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        if os.path.isfile(filename):
            if use_gpu:
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location='cpu')

            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model.share_memory()
            optimizer.load_state_dict(checkpoint['optimizer'])

            if use_gpu:
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

        model_checkpoint_filename = "{}_{:04d}.pth.tar".format(
            self.model_file_base_name, epoch)
        filename = os.path.join(self.models_dir, model_checkpoint_filename)

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

        self.model_file_names.append(filename)

    def evaluate_accuracy(self, loader, loader_iter, set_size):
    # def evaluate_accuracy(self, loader, set_size):
        """
        Returns accuracy of model determined from test images
        """

        start_test_time = time.time()
        self.model.eval()
        self.model.cuda()

        if self.print:
            print('eval time: {}'.format(time.time() - start_test_time))

        overall_acc = 0.0
        class_correct = list(0.0 for i in range(self.num_classes))
        class_total = list(0.0 for i in range(self.num_classes))
        class_acc = list(0.0 for i in range(self.num_classes))
        total_gpu_transfer_time = 0

        # Iterate over the test loader
        # for images, labels in loader:
        for i in range(len(loader)):
            images, labels = next(loader_iter)

            if self.cuda_avail:
                start_transfer_time = time.time()
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                total_gpu_transfer_time += time.time() - start_transfer_time

            # Predict classes using images from the test set
            outputs = self.model(images)

            # Pick max prediction
            _, prediction = torch.max(outputs.data, 1)

            # Compare to actual class to obtain accuracy
            overall_acc += torch.sum(prediction == labels.data).float()
            c = (prediction == labels.data)

            for i in range(len(c)):
                label = labels.data[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(self.num_classes):
            if class_total[i] > 0:
                if self.print:
                    print('Accuracy of {} mugs: {}'.format(i+1, class_correct[i]/class_total[i]), flush=True)
                class_acc[i] = class_correct[i]/class_total[i]

        if self.print:
            print('total_gpu_transfer_time (test): {}'.format(total_gpu_transfer_time))
            print('test_time: {}'.format(time.time() - start_test_time))

        # Compute the average acc and loss over all test images
        overall_acc = overall_acc / set_size

        return overall_acc, class_acc

    def train(self, num_epochs):
        """
        Trains model with num_epochs epochs.
        """

        print('training with {} epochs'.format(num_epochs), flush=True)

        model_accuracies_csv = os.path.join(self.models_dir, '{}.csv'.format(self.model_file_base_name))
        f = open(model_accuracies_csv, 'w')
        if self.using_counterexample_set:
            f.write('epoch, training_loss, training_acc, test_acc, counterex_acc, is_new_best, test_class_1, test_class_2, test_class_3, test_class_4, test_class_5, counterex_class_3,\n')
        else:
            f.write('epoch, training_loss, training_acc, test_acc, is_new_best, test_class_1, test_class_2, test_class_3, test_class_4, test_class_5,\n')
        f.flush()

        best_acc = 0.0
        # print('starting to create loader iters', flush=True)
        test_loader_iter = itertools.cycle(self.test_loader)
        # print('created test loader', flush=True)
        # training_loader_iter = itertools.cycle(self.train_loader)
        # # training_loader_iter = iter(self.train_loader)
        # # test_loader_iter = iter(self.test_loader)
        # print('finished creating loader iters', flush=True)
        # training_loader_iter = iter(self.train_loader)
        training_loader_arr = []

        for images, labels in self.train_loader:
            training_loader_arr.append((images, labels))
        if self.print:
            print('finished creating training_loader_arr', flush=True)

        for epoch in range(num_epochs):
            if self.print:
                print('------------------------', flush=True)
            start_epoch_time = time.time()
            total_gpu_transfer_time = 0
            training_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            time_in_training_loop = 0.0
            time_loading = 0.0

            # for i in range(len(self.train_loader)):
            # for images, labels in self.train_loader:
                # images, labels = next(training_loader_iter)

            rand_arr = random.sample(range(0, len(self.train_loader)), len(self.train_loader))

            for i in rand_arr:
                images, labels = training_loader_arr[i]

                start_training_time = time.time()

                time_loading += time.time() - start_training_time

                self.model.train()
                # Move images and labels to gpu if available
                if self.cuda_avail:
                    start_transfer_time = time.time()
                    images = Variable(images.cuda(0))
                    labels = Variable(labels.cuda(0))
                    total_gpu_transfer_time += time.time() - start_transfer_time
                # print(labels.device)

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

                train_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_acc += torch.sum(prediction == labels.data).float()
                time_in_training_loop += time.time() - start_training_time
            
            training_time = time.time() - training_time

            if self.print:
                print('time_loading training: {}'.format(time_loading), flush=True)
                print('total_gpu_transfer_time (training): {}'.format(total_gpu_transfer_time), flush=True)
                print('time in training loop: {}'.format(time_in_training_loop), flush=True)
                print('total training time: {}'.format(training_time), flush=True)

            # Call the learning rate adjustment function
            self.adjust_learning_rate(epoch)

            # Compute the average acc and loss over all training images
            train_acc = train_acc / self.training_set_size
            train_loss = train_loss / self.training_set_size

            # Evaluate on the test set
            # print('test acc', flush=True)
            test_acc, test_class_accs = self.evaluate_accuracy(self.test_loader, test_loader_iter, self.test_set_size)
            # test_acc, test_class_accs = self.evaluate_accuracy(self.test_loader, self.test_set_size)
            
            if self.using_counterexample_set:
                # print('counterex acc', flush=True)
                counterexample_acc, counterexample_class_accs = self.evaluate_accuracy(self.counterexample_loader, self.counterexample_set_size)

            # Print the metrics
            # print("Epoch {}, Train Accuracy: {}, Train Loss: {}, Test Accuracy: {},"
            #     "Counterexample Accuracy: {}".format(
            #     epoch, train_acc, train_loss, test_acc, counterexample_acc))

            # self.train_accuracies.append(train_acc)
            # self.test_accuracies.append(test_acc)
            # self.counterexample_accuracies.append(counterexample_acc)

            if self.using_counterexample_set:
                f.write('{:3d}, {:2.5f}, {:1.5f}, {:1.5f}, {:1.5f}, {:1d}, {:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}, {:1.5f},\n'.format(
                    epoch, train_loss, train_acc, test_acc, counterexample_acc, test_acc > best_acc,
                    test_class_accs[0], test_class_accs[1], test_class_accs[2], test_class_accs[3], test_class_accs[4],
                    counterexample_class_accs[2]))
            else:
                f.write('{:3d}, {:2.5f}, {:1.5f}, {:1.5f}, {:1d}, {:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}, {:1.5f},\n'.format(
                    epoch, train_loss, train_acc, test_acc, test_acc > best_acc,
                    test_class_accs[0], test_class_accs[1], test_class_accs[2], test_class_accs[3], test_class_accs[4]))
            f.flush()

            # Save the model if the test acc is greater than our current best
            if test_acc > best_acc:
                self.save_checkpoint(epoch)
                best_acc = test_acc
                print("New best acc is {}, epoch {}".format(best_acc, epoch), flush=True)

            if self.print:
                print('epoch time: {}'.format(time.time() - start_epoch_time), flush=True)

        # Move last epoch to current model and delete all other models
        model_file_name = os.path.join(self.models_dir, '{}.pth.tar'.format(self.model_file_base_name))

        print('copying {} to {}'.format(self.model_file_names[-1], model_file_name), flush=True)
        shutil.copy(self.model_file_names[-1], model_file_name)

        for model_file_name in self.model_file_names:
            os.remove(model_file_name)

        # print('train_accuracies: {}'.format(self.train_accuracies))
        # print('test_accuracies: {}'.format(self.test_accuracies))
        # print('counterexample_accuracies: {}'.format(self.counterexample_accuracies))

        f.close()

# from simple_net import SimpleNet

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
import requests
import shutil
from io import open, BytesIO
import os
from PIL import Image, ImageFile
import json
import numpy as np

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels,
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

path = "mug_numeration_54.model"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model = SimpleNet(num_classes=5)
model.load_state_dict(checkpoint)
model.eval()
print(model)

def predict_image(image_path, num_mugs):
    print('in predict_image image_path: {}'.format(image_path))

    image = Image.open(image_path)
    image = image.convert('RGB')

    max_edge_length = 369   # TODO find this programatically

    # Define transformations for the image
    transformation = transforms.Compose([
        transforms.CenterCrop((max_edge_length)),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()
    print('image_tensor: ', image_tensor)

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    # Add a softmax layer to extract probabilities
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print('probabilities: {}'.format(probabilities.data.numpy()))

    print('output data: ', output.data.numpy())
    index = output.data.numpy().argmax()

    classes = [1, 2, 3, 4, 5]

    word = 'are'
    if classes[index] == 1:
        word = 'is'

    print('there {} {} mugs'.format(word, classes[index]))

    if classes[index] != num_mugs:
        print('WRONG, the actual number of mugs is {}!'.format(num_mugs))
    else:
        print('this is correct')

    return index

def main():
    imagefile = "5_4_color.png"     # 2
    # imagefile = "4_1038_color.png"
    # imagefile = "5_1259_color.png"

    imagepath = os.path.join(os.getcwd(), imagefile)

    print('imagepath: {}'.format(imagepath))

    # Run prediction function and obtain predicted class index
    index = predict_image(imagepath, num_mugs=int(imagefile[0]))
    print('index: {}'.format(index))

if __name__ == "__main__":
    main()
    
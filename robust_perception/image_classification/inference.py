from ..image_classification.simple_net import SimpleNet

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

correct_num = 0
wrong_num = 0

def predict_image(model, image_path, num_mugs):
    print('in predict_image image_path: {}'.format(image_path))

    image = Image.open(image_path)
    image = image.convert('RGB')

    # max_edge_length = 369   # TODO find this programatically

    # Define transformations for the image
    transformation = transforms.Compose([
        # transforms.CenterCrop((max_edge_length)),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()
    # print('image_tensor: ', image_tensor)

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a Variable
    input = Variable(image_tensor.cuda())

    # Predict the class of the image
    output = model(input)

    # Add a softmax layer to extract probabilities
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print('probabilities: {}'.format(probabilities.data.cpu().numpy()))

    # print('output data: ', output.data.numpy())
    index = output.data.cpu().numpy().argmax()

    classes = [1, 2, 3, 4, 5]

    word = 'are'
    s = 's'
    if classes[index] == 1:
        word = 'is'
        s = ''

    print('there {} {} mug{}'.format(word, classes[index], s))

    if classes[index] != num_mugs:
        # print('WRONG, the actual number of mugs is {}!'.format(num_mugs))
        global wrong_num
        wrong_num += 1
    else:
        # print('this is correct')
        global correct_num
        correct_num += 1

    return index

def main():
    # path = '/home/maggiewang/Workspace/robust_perception/robust_perception/data/retrained_with_counterexamples/cma_es/models/mug_numeration_classifier_003.pth.tar'
    path = '/home/maggiewang/Workspace/robust_perception/robust_perception/data/retrained_with_counterexamples/cma_es/models/mug_numeration_classifier_000.pth.tar'
    checkpoint = torch.load(path)

    model = SimpleNet(num_classes=5)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    # print(model)

    # imagefile = "3_62_color.png"
    # imagefile = "0082_3_color.png"
    # imagefile = "1_0_color.png"     # 2
    # imagefile = "4_1038_color.png"
    # imagefile = "5_1259_color.png"

    # fix this gross method
    # path = '/home/maggiewang/Workspace/robust_perception/robust_perception/dataset_generation/images/classification_clean/training_set/3'
    # path = '/home/maggiewang/Workspace/robust_perception/robust_perception/optimization/data_rbfopt'

    correct_files = []

    for i in range(1, 6):
        path = '/home/maggiewang/Workspace/robust_perception/robust_perception/data/retrained_with_counterexamples/cma_es_incorrectlycopiedcounterexs/counterexample_set/{}'.format(i)
        for file in os.listdir(path):
            # print(file)
            if file.endswith(".png"):
                # print('file: {}'.format(file))
                index = predict_image(model, os.path.join(path, file), num_mugs=i)
                # if index == 2:
                    # correct_files.append(file)

    # print('correct_files:', correct_files)
    # print(len(correct_files))

    print('correct_num: {}, wrong_num: {}'.format(correct_num, wrong_num))
    print('percent corr: {}'.format(1.0 * correct_num/(wrong_num+correct_num)))

    # imagepath = os.path.join(os.getcwd(), imagefile)

    # print('imagepath: {}'.format(imagepath))

    # Run prediction function and obtain predicted class index
    # index = predict_image(model, imagepath, num_mugs=int(imagefile[0]))
    # print('index: {}'.format(index))

if __name__ == "__main__":
    main()
    
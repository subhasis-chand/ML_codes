import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30 * 40, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.fc4 = nn.Linear(1200, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        # self.fc6 = nn.Linear(600, 1200)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        # x = torch.sigmoid(self.fc5(x))
        # x = torch.sigmoid(self.fc6(x))
        return x

model = torch.load('./linearModelFinal')
model.eval()
print(model)

trainPathInput = '/Users/subhasis/myWorks/dataCone/trainingSet/obstacles/obstacles/'
trainingInputList = []
for file in os.listdir(trainPathInput):
    if file[-4:] == '.png':
        trainingInputList.append(file)
trainingInputList = sorted(trainingInputList)

stackedIn = np.empty((200, 1200))
for i in range(0, 200):
    path = trainPathInput + trainingInputList[i]
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (40, 30), interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (1, 1200))
    stackedIn[i-0, :] = img / 255.
imagesIn = torch.from_numpy(stackedIn).float()

outputs = model(imagesIn)

print('output shape: ', outputs.shape)

displayPred = outputs.detach().numpy()
displayPred = np.reshape(displayPred[56], (30, 40))
plt.imsave('output.png', displayPred)


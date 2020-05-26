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

model = Net()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

criterion = nn.MSELoss()
def crossEntropy(predicted, actual):
    loss = torch.log(1-((predicted - actual)**2))
    # print("No of ones:", ((predicted-actual)==0).sum())
    # print("Loss: ", -loss.sum())
    criterion = nn.MSELoss()
    return -loss.sum() # + criterion(predicted, actual)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#########  Initialising Constants  ##########

trainingBatchSize = 200
testingBatchSize = 10
n_epochs = 125
trainPathInput = '/Users/subhasis/myWorks/dataCone/trainingSet/obstacles/obstacles/'
trainPathOutput = '/Users/subhasis/myWorks/dataCone/trainingSet/obs_n_path/obs_n_path/'
testPathInput = '/Users/subhasis/myWorks/dataCone/testingSet/obstacles/obstacles/'
testPathOutput = '/Users/subhasis/myWorks/dataCone/testingSet/obs_n_path/obs_n_path/'

#########  Building the Data Loader  ##########
trainingInputList = []
for file in os.listdir(trainPathInput):
    if file[-4:] == '.png':
        trainingInputList.append(file)
trainingInputList = sorted(trainingInputList)

trainingOutputList = []
for file in os.listdir(trainPathOutput):
    if file[-4:] == '.png':
        trainingOutputList.append(file)
trainingOutputList = sorted(trainingOutputList)

testingInputList = []
for file in os.listdir(testPathInput):
    if file[-4:] == '.png':
        testingInputList.append(file)
testingInputList = sorted(testingInputList)

testingOutputList = []
for file in os.listdir(testPathOutput):
    if file[-4:] == '.png':
        testingOutputList.append(file)
testingOutputList = sorted(testingOutputList)

testImg = cv2.imread(trainPathInput+trainingInputList[0])
testImg = cv2.resize(testImg, (40, 30)) 
row, col = testImg.shape[0], testImg.shape[1]
# plt.imshow(testImg)
# plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)


#########  Training  ##########
####### This reproduces the input images properly after the training....
for epoch in range(1 , n_epochs+1):
    train_loss = 0.0
    startingIndex = 0
    endingIndex = trainingBatchSize

    while(1):
        if endingIndex > len(trainingInputList):
            break
        print(startingIndex, '-', endingIndex)

        stackedIn = np.empty((trainingBatchSize, row * col))
        
        for i in range(startingIndex, endingIndex):
            path = trainPathInput + trainingInputList[i]
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (40, 30), interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, (1, row * col))
            stackedIn[i-startingIndex, :] = img / 255.
        imagesIn = torch.from_numpy(stackedIn).float()

        stackedOut = np.empty((trainingBatchSize, row * col))
        
        for i in range(startingIndex, endingIndex):
            path = trainPathOutput + trainingOutputList[i]
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (40, 30), interpolation=cv2.INTER_CUBIC) 
            img = np.reshape(img, (1, row * col))
            stackedOut[i-startingIndex, :] = img / 255.
        imagesOut = torch.from_numpy(stackedOut).float()

        startingIndex = endingIndex
        endingIndex += trainingBatchSize

        optimizer.zero_grad()
        outputs = model(imagesIn)

        if startingIndex > 78000:
            displayIn = imagesIn.detach().numpy()
            displayIn = np.reshape(displayIn[0], (30, 40))
            displayPred = outputs.detach().numpy()
            displayPred = np.reshape(displayPred[0], (30, 40))
            displayActual = imagesOut.detach().numpy()
            displayActual = np.reshape(displayActual[0], (30, 40))
            diff = (displayPred - displayActual)
            diff = diff-(np.min(diff))
            plt.imsave('epoch_' + str(epoch) + 'img_' + str(startingIndex) + 'pred.png', displayPred)
            plt.imsave('epoch_' + str(epoch) + 'img_' + str(startingIndex) + 'Actual.png', displayActual)
            print('image saved...')

        loss = crossEntropy(outputs, imagesOut)
        # loss = criterion(outputs, imagesOut)
        print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*imagesIn.size(0)

            
    train_loss = train_loss/len(trainingInputList)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    if epoch == 50 or epoch == 75 or epoch == 100:
        torch.save(model, './linearModelEpoch' + str(epoch))

torch.save(model, './linearModelFinal')
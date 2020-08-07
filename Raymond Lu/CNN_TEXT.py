#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import re

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe

from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable



import numpy as np
import sklearn

from sklearn.feature_extraction import text as sktext

DIMENSION = 50

stopWords = set(sktext.ENGLISH_STOP_WORDS)


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    new_list = []
    # cop = re.compile("[^a-zA-Z0-9]")
    cop = re.compile("[^a-z0-9]")
    # cleanr = re.compile('<.*?>')
    for i in sample:
        i = i.lower()
        i = re.sub(r'http\S+', '',i)
        i = cop.sub(' ', i)
        # i = re.sub(cleanr, ' ', i)
        # i = re.sub(r'[?|!|\'|"|#|&|^]',r' ',i)
        # i = re.sub(r'[.|,|)|(|\|/]',r' ',i)
        if len(i) > 1 and i not in stopWords:
            new_list.append(i)


    return new_list


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch


wordVectors = GloVe(name='6B', dim=DIMENSION)


###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    # label = datasetLabel.view((1,-1))
    datasetLabel = Variable(torch.LongTensor(datasetLabel)).to(device)
    return datasetLabel


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # temp = netOutput.view([-1])
    # for i in range(len(temp)):
    #     if temp[i] < 1.5:
    #         temp[i] = 1.
    #     elif temp[i] < 2.5:
    #         temp[i] = 2.
    #     elif temp[i] < 3.5:
    #         temp[i] = 3.
    #     elif temp[i] < 4.5:
    #         temp[i] = 4.
    #     else:
    #         temp[i] = 5.
    # return temp
    netOutput = Variable(torch.FloatTensor(netOutput)).to(device)
    temp = netOutput.view([-1]) 
    for i in range(len(temp)):
        temp[i] += 1
    return (temp)


###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        self.window_sizes=(3, 4, 5)
        self.num_filters = 128


        self.convs = tnn.ModuleList([
            tnn.Conv2d(1, self.num_filters, [window_size, DIMENSION], padding=(window_size - 1, 0))
            for window_size in self.window_sizes])

        self.fc = nn.Linear(self.num_filters * len(self.window_sizes), 5)


    def forward(self, input, length):
        #packed sequence
        input = torch.unsqueeze(input, 1)
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(input))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]
        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]
        # Prediction
        probs = F.softmax(logits)       # [B, class]
        # classes = torch.max(probs, 1)[1]# [B]
        # return classes.data[0]
        return probs

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.crtic = tnn.MSELoss()

    def forward(self, output, target):
        return self.crtic(output, target)


net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
# lossFunc = tnn.MSELoss()
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
# optimiser = toptim.SGD(net.parameters(), lr=0.2)
optimizer = optim.Adam(net.parameters(), lr=0.001)

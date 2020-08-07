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

import csv



import numpy as np
import sklearn

from sklearn.feature_extraction import text as sktext

DIMENSION = 50
wordVectors = GloVe(name='6B', dim=DIMENSION)

stopWords = set(sktext.ENGLISH_STOP_WORDS)


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    new_list = []
    # cop = re.compile("[^a-zA-Z0-9]")
    cop = re.compile("[^a-z0-9|^\'|^_|^\-|\s|^(\d+ \. \d+)|^$|(\-\-)]")
    # cleanr = re.compile('<.*?>')
    for i in sample:
        i = i.lower()
        i = re.sub(r'http\S+', '',i)
        i = cop.sub('', i)
        if len(i) > 1 and i not in stopWords:
            new_list.append(i)


    return new_list


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch


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
    # datasetLabel = Variable(torch.LongTensor(datasetLabel))
    datasetLabel = datasetLabel.long()
    # print("Now the datasetLabel's type is ",datasetLabel.type())
    print("Now the data label is: ",datasetLabel.data)
    # datasetLabel = Variable(torch.LongTensor(datasetLabel))
    return datasetLabel


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """

    netOutput = netOutput.float()
    temp = netOutput.view([-1]) 
    # print("In converNetOutput now, the temp is:",temp)
    # for i in range(len(temp)):
    #     temp[i] += 1
    return (temp)


###########################################################################
################### The following determines the model ####################
###########################################################################

#------ Util Functions--------#
def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, tnn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return torch.sum(output, 0).unsqueeze(0)

class WordAttNet(tnn.Module):
    def __init__(self, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict_len, embed_size = wordVectors.vectors.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = tnn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = tnn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = tnn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = tnn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = tnn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output, h_output

class SentAttNet(tnn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=5):
        super(SentAttNet, self).__init__()
        self.sent_weight = tnn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = tnn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = tnn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
        self.gru = tnn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = tnn.Linear(2 * sent_hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)
        return output, h_output



class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    def __init__(self):
        super(HierAttNet, self).__init__()
        self.batch_size = batchSize
        self.word_hidden_size = 50
        self.sent_hidden_size = 50
        self.word_att_net = WordAttNet(word_hidden_size)
        self.sent_att_net = SentAttNet()
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input,length):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        return output

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
# lossFunc = F.nll_loss
###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
# optimiser = toptim.SGD(net.parameters(), lr=0.2)
# optimiser = toptim.Adam(net.parameters(), lr=0.001)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9)

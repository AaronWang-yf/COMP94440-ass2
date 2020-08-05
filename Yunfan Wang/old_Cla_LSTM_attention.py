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
from torch.autograd import Variable
from torchtext.vocab import GloVe


# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    new_list = []
    cop = re.compile("[^a-zA-Z\s\d]")
    for i in sample:
        i = cop.sub(' ', i)
        if len(i) > 1 and i not in stopWords:
            new_list.append(i)
    return new_list


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch


stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can", "couldn", "d", "did", "do", "does", "doesn", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
             "have", "haven", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
             "if", "in", "into", "is", "isn", "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's", "should", "should've", "shouldn", "so", "some",
             "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
             "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with", "won", "wouldn",
             "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]
wordVectors = GloVe(name='6B', dim=100)


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
    # class1 = torch.Tensor([1., 0., 0., 0., 0.])
    # class2 = torch.Tensor([0., 1., 0., 0., 0.])
    # class3 = torch.Tensor([0., 0., 1., 0., 0.])
    # class4 = torch.Tensor([0., 0., 0., 1., 0.])
    # class5 = torch.Tensor([0., 0., 0., 0., 1.])
    # class_dict = {1:class1,2:class2,3:class3,4:class4,5:class5}
    # print(len(datasetLabel))
    # for i in range(len(datasetLabel)):
    #     print(int(datasetLabel[i]))
    #     datasetLabel[i] = class_dict[int(datasetLabel[i])]
    datasetLabel = datasetLabel.long() - 1
    return datasetLabel


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    temp = torch.max(netOutput, 1)[1]
    temp += 1
    return temp


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

        self.embed_dim = 100  # dim of glove
        self.hidden = 256

        self.batch_size = 32
        self.output_size = 5
        self.hidden_size = 256
        self.bidirectional = True
        self.dropout = 0.5

        self.layer_size = 2

        self.lstm = tnn.LSTM(self.embed_dim,
                             self.hidden_size,
                             self.layer_size,
                             dropout=self.dropout,
                             bidirectional=self.bidirectional
                             )
        self.init_w = Variable(torch.Tensor(1, 2 * self.hidden), requires_grad=True)
        self.init_w = tnn.Parameter(self.init_w)

        self.fc1 = tnn.Linear(self.hidden * 2, self.hidden)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.fc2 = tnn.Linear(self.hidden, 5)
        # self.attention_size = 20
        # self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
        # self.u_omega = Variable(torch.zeros(self.attention_size).cuda())

    # def attention_net(self, lstm_output, length):
    #     output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
    #     attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
    #     attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
    #     exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, length])
    #     alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
    #     alphas_reshape = torch.Tensor.reshape(alphas, [-1, length, 1])
    #     state = lstm_output.permute(1, 0, 2)
    #     attn_output = torch.sum(state * alphas_reshape, 1)
    #     return attn_output

    def forward(self, input, length):
        # embeded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        h_0 = Variable(torch.zeros(self.layer_size, len(length), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.layer_size, len(length), self.hidden_size).cuda())
        input = input.permute(1, 0, 2)
        output, (hidden, cekk) = self.lstm(input, (h_0, c_0))
        # attn_output = self.attention_net(lstm_out, length[0])
        # logits = self.fc2(attn_output)
        M = torch.matmul(self.init_w, output.permute(1, 2, 0))
        alpha = tnn.functional.softmax(M, dim=0)
        out = torch.matmul(alpha, output.permute(1, 0, 2)).squeeze()
        out = self.fc1(out)
        probs = tnn.functional.log_softmax(self.fc2(out), dim=1)
        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # output = self.unpack(output, batch_first=True)
        # dense_outputs = self.fc(hidden[-1, :, :])
        # dense_outputs = dense_outputs.view([-1])
        # probs = tnn.functional.log_softmax(dense_outputs, dim=1)
        return probs


# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#         self.critic = tnn.MSELoss()
#
#     def forward(self, output, target):
#         return self.critic(output, target)


net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = tnn.NLLLoss()

###########################################################################
################ The following determines training options ################
###########################################################################
trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.002)

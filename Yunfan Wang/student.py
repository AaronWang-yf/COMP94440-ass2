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
    cop = re.compile("[^a-zA-Z0-9]")
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
             "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can",
             "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don",
             "don't", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't",
             "have", "haven",
             "haven't", "having", "he", "her", "here", "hers", "herself",
             "him", "himself", "his", "how", "i", "if", "in", "into", "is",
             "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't",
             "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should",
             "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn",
             "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]
wordVectors = GloVe(name='6B', dim=50)


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
    return datasetLabel


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    temp = netOutput.view([-1])
    for i in range(len(temp)):
        if temp[i] < 1.5:
            temp[i] = 1.
        elif temp[i] < 2.5:
            temp[i] = 2.
        elif temp[i] < 3.5:
            temp[i] = 3.
        elif temp[i] < 4.5:
            temp[i] = 4.
        else:
            temp[i] = 5.
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

        self.rnn = tnn.LSTM(
            input_size=50,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )

        self.fc = tnn.Linear(512, 1)
        self.unpack = tnn.utils.rnn.pad_packed_sequence

    def forward(self, input, length):
        embeded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        output, (hidden, cell) = self.rnn(embeded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # output=self.unpack(output, batch_first=True)
        dense_outputs = self.fc(hidden)
        dense_outputs = dense_outputs.view([-1])
        return dense_outputs


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
lossFunc = tnn.MSELoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.2)

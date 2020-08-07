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


trainValSplit = 0.8
batchSize = 32
epochs = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

DIMENSION = 100

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
        dict_len, embedding_dim = wordVectors.vectors.shape
        self.hidden_size = 128 
        self.num_classes = 5 
        self.vocab_size = dict_len 
        self.embedding_dim = embedding_dim 
        self.embed = tnn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim).to(device)
        self.rnn = tnn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True).to(device)
        self.hidden2label = tnn.Linear(self.hidden_size, self.num_classes).to(device)

    def forward(self, input, length):
        h0 = torch.randn(1, batch_size, self.hidden_size).to(device)
        _, hn = self.rnn(input, h0)
        logits = self.hidden2label(hn).squeeze(0)
        return logits



net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
# lossFunc = tnn.NLLLoss()
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

optimiser = toptim.Adam(net.parameters(), lr=0.001)
# optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.002, momentum=0.9)

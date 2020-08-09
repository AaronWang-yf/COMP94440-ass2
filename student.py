"""
UNSW 20T2 COMP9444 Project 2
Group Name: Galactic Repairmen 
Gruop Member: Minrui Lu, Yunfan Wang 
Group ID: g022233
Weighted Score on Vlab (CPU mode): 63.66
"""

"""
DESCRIPTION:

First we got a list of stop words from: https://gist.github.com/sebleier/554280. After loading 
the raw text data and tokenizing it, we remove punctuations and non-English characters in the pre-processing and 
non-negation stop words are discarded from the tokenized words in the construction of data field.
We assume negation words are strongly associated with the ranking because
they are significant components of sentiment expression.

In the training part, we build an LSTM (Long Short Term Memory) RNN (Recurrent Neural Network) model 
because we reckon LSTM can extract text features from temporal domain, which, compared to CNN, fits
closer to the way of human thinking. Additionally, LSTM processes the entire sequence of data, and in
the scenario of this project, the sequence of data is a sentence which requires us to understand it as
a whole instead of interpretting each word in it one by one. With forget gates, input gates and output
gates, LSTM can decide what kinds of information from previous words are passed to their successors. And
this contributes to finding the relations between words, and the relations between words and ranking.
During the training process, we transform the label to LongTensor type and subtract value one from each label due
to the requirements of NLL Loss fucntion in PyTorch that training labels should commence from 0.

In the evaluation part, we enable the convertNetOutput function to find out the predicted labels with the maximum 
probabilities and add them with one offset so that the range of the output label will be transformed back to 
{1.0, 2.0, 3.0, 4.0, 5.0}.

In our experiments, we find that setting the LSTM to be bidirectional is not significant in enhancing 
the prediction performance and we decide to set it to be single-directional for faster execution. When it
comes to the hidden size, if we set the hidden size to be 512, the output model will be 
overfitted for the training data, while, if the hidden size is 128, the weighted score will drop by
approximately 2 scores. In addition, we have tried to construct LSTM regression model for this dataset, 
but its results, on average, are 2 scores lower than the current one. As for metaparameters and other 
global parameters after a series of trials,  we find that using epoch value of 6 can get
the optimal result because under this situation the model is not overfitted. Moreover, settiing a dropout rate of 0.4 can get 
a nice convergence result with overfitting. We have tried the combination of softmax 
activation function and cross entropy loss function, and its result is lower than 60 scores. Accordingly, we do not 
consider that. Speaking of the number of layers, if we set it to be 1, the score will decrease by 2-3 scores. And if it is 3, 
there is no significant difference. When the number is set to be more than 4, due to the increase of complexity, the gradient 
vanishes and the training fails. The number of drop out proportion does not affect the final score significantly.

"""

import re
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe

###########################################################################
###################### Self Define Parameters##############################
###########################################################################
stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can", "d", "did", "do", "does", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "has", 
             "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
             "if", "in", "into", "is",
            "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's", "should", "should've", "so", "some",
             "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "we", "were", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with",
             "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]

DIMENSION  = 100
wordVectors = GloVe(name='6B', dim=DIMENSION)

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

"""
Remove punctuations, single characters and any non-English characters
"""
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    new_list = []
    cop = re.compile("[^a-zA-Z\s\d]")
    for i in sample:
        i = cop.sub(' ', i)
        if (len(i) > 1):
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
"""
Because the NLL Loss function in Pytorch requires the input tensor is of Long type, and the classfication of NLL Loss starts from 0. We transform the 
type of input to Long type and minus 1 from the input label to guarantee the classification range is correct.
"""
def convertLabel(datasetLabel):
    datasetLabel = datasetLabel.long() - 1
    return datasetLabel


"""
torch.max() is used to find the label with the maximum probablity for each line in the netOutput. Since the 
net output ranges from 0 to 4, we need to add a 1 padding back to get the correct prediction.
"""
def convertNetOutput(netOutput):
    temp = torch.max(netOutput,1)[1]
    temp+=1
    return temp.float()


###########################################################################
################### The following determines the model ####################
###########################################################################

"""
An LSTM RNN  model is utilized in the network
"""

class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()

        self.rnn = tnn.LSTM(
            input_size=DIMENSION,
            hidden_size=256,
            num_layers=2,
            bidirectional=False,
            dropout=0.4,
            batch_first=True
        )
        self.num_classes = 5 # 5 different ranks in the given dataset
        self.fc = tnn.Linear(256, self.num_classes)

    def forward(self, input, length):
        # Reshape the input to get it fit in the LSTM network
        # We transform the input's size from [batchSize,length,DIMENSION] to [batchSize*length,DIMENSION],
        # and then put it into LSTM model.
        embeded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        output, (hidden, cell) = self.rnn(embeded)
        # Get the last layer of hidden unit and pass it to the fully connected layer
        dense_outputs = self.fc(hidden[-1, :, :])
        probs = tnn.functional.log_softmax(dense_outputs, dim=1)
        return probs


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
epochs = 6
optimiser = toptim.Adam(net.parameters(), lr=0.002)

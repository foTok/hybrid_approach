"""
defines some loss function
"""

import torch
from torch.autograd import Variable

def CrossEntropy(output, label):
    """
    output: pytorch Variable vector, output of nerual network
    label: pytorch Variable vector, observed values
    warning: make sure all numbers in output and label in the range [0, 1]
    """
    assert len(output) == len(label)
    loss = Variable(torch.FloatTensor([0]*len(output)))
    for o, l ,i in zip(output, label, range(len(output))):
        loss[i] = -(l*torch.log(o+1e-10) + (1-l)*torch.log(1-o+1e-10))
    loss0 = torch.sum(loss)
    return loss0

"""
defines some loss function
"""

import torch
from torch.autograd import Variable

def CrossEntropy(output, label, size_average = True):
    """
    output: pytorch Variable vector, output of nerual network
    label: pytorch Variable vector, observed values
    warning: make sure all numbers in output and label in the range [0, 1]
    output, label: batch × fault
    """
    assert output.size() == label.size()
    loss = -(label*torch.log(output+1e-10) + (1-label)*torch.log(1-output+1e-10))
    loss0 = torch.sum(loss, dim = 1)
    if size_average:
        loss1 = torch.mean(loss0)
    else:
        loss1 = torch.sum(loss0)
    return loss1

def MSE(output, label, size_average = True):
    """
    output: pytorch Variable vector, output of nerual network
    label: pytorch Variable vector, observed values
    output, label: batch × fault
    """
    assert output.size() == label.size()
    loss = (label - output)**2
    loss0 = torch.sum(loss, dim = 1)
    if size_average:
        loss1 = torch.mean(loss0)
    else:
        loss1 = torch.sum(loss0)
    return loss1

def PRO(output, label, size_average = True):
    """
    output: pytorch Variable vector, output of nerual network
    label: pytorch Variable vector, observed values
    output: batch × 2R
    label: batch × R
    """
    assert len(output) == len(label)
    mu = output[:, 0]
    sigma = output[:, 1]
    loss = ((label - mu) / (sigma + 1e-10)) + 0.01*sigma
    loss0 = loss**2
    if size_average:
        loss1 = torch.mean(loss0)
    else:
        loss1 = torch.sum(loss0)
    return loss1

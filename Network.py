#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class NaiveNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, learning_rate, gpu=False):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)
        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            print 'Transferring network to GPU...'
            self.cuda()
            print 'Network transferred.'

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y

    def learn(self, x, target):
        x = torch.from_numpy(x)
        target = torch.from_numpy(target)
        if self.gpu:
            x = x.cuda()
            target = target.cuda()
        x = Variable(x)
        target = Variable(target)

        y = self.forward(x)
        loss = self.criterion(y, target)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().data.numpy(), self.output_transfer(y).cpu().data.numpy()

    def output_transfer(self, y):
        _, predicted_label = torch.max(y, 1)
        return predicted_label

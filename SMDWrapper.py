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

class SMDWrapper:
    def __init__(self, net, criterion, learning_rate):
        self.net = net
        self.lr = learning_rate
        self.epsilon = 1e-4
        self.criterion = criterion
        self.p_dict = dict()
        self.v_dict = dict()
        self.v_delta_dict = dict()
        self.delta_dict = dict()
        for param in self.net.parameters():
            self.p_dict[param] = torch.ones(param.data.size()) * self.lr
            self.v_dict[param] = torch.zeros(param.data.size())
            self.v_delta_dict[param] = torch.zeros(param.data.size())
            self.delta_dict[param] = torch.zeros(param.data.size())

        if self.net.gpu:
            for info_dict in [self.p_dict, self.v_dict, self.v_delta_dict, self.delta_dict]:
                for param in info_dict.keys():
                    info_dict[param] = info_dict[param].cuda()

    def gradient(self, x, target):
        y = self.net.forward(x)
        loss = self.criterion(y, target)
        self.net.zero_grad()
        loss.backward()
        return loss, y

    def learn(self, x, target):
        x = torch.from_numpy(x)
        target = torch.from_numpy(target)
        if self.net.gpu:
            x = x.cuda()
            target = target.cuda()
        x = Variable(x)
        target = Variable(target)

        loss, y = self.gradient(x, target)
        for param in self.net.parameters():
            self.delta_dict[param].copy_(-param.grad.data)
            self.p_dict[param].mul_(torch.exp(self.lr * self.delta_dict[param] * self.v_dict[param]))
            param.data.add_(self.epsilon * self.v_dict[param])
            self.gradient(x, target)
            hessian_vector = (param.grad.data + self.delta_dict[param]) / self.epsilon
            self.v_delta_dict[param].copy_(self.p_dict[param] * (self.delta_dict[param] - hessian_vector))
            param.data.sub_(self.epsilon * self.v_dict[param])
        for param in self.net.parameters():
            param.data.add_(self.p_dict[param] * self.delta_dict[param])
            self.v_dict[param].add_(self.v_delta_dict[param])
        return loss.cpu().data.numpy(), self.net.output_transfer(y).cpu().data.numpy()

if __name__ == '__main__':
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(3, 1, bias=False)

        def forward(self, x):
            return self.fc1(x)

        def output_transfer(self, y):
            return y

    net = SimpleNet()
    for ind, param in enumerate(net.parameters()):
        param.data.copy_(torch.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype='float32')))
    x = np.array([[0.1, 0.2, 0.3]], dtype='float32')
    target = np.array([1.0], dtype='float32')
    criterion = nn.MSELoss()
    smd = SMDWrapper(net, criterion, 0.1)
    print smd.learn(x, target)

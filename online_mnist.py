import numpy as np
import pickle
import logging
from load_mnist import *
from SMDWrapper import *
from Network import *

tag = 'online_MNIST'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/%s.txt' % tag)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

datapath = './dataset/mnist'
train_x, train_y = load_mnist('training', path=datapath)

dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10

train_examples = 5000
train_x = train_x[: train_examples, :].reshape([-1, dim_in])
train_y = np.argmax(train_y[: train_examples, :], axis=1)

def bp_network(lr):
    return NaiveNet(dim_in, dim_hidden, dim_out, lr, gpu=True)

def smd_network(lr):
    net = bp_network(lr)
    smd = SMDWrapper(net, net.criterion, lr)
    return smd

methods = [bp_network, smd_network]
# methods = [bp_network]
# methods=  [smd_network]

window_size = 100
train_acc = np.zeros((len(methods), train_examples))
def train(learning_rate):
    learners = [method(learning_rate) for method in methods]
    for train_index in range(train_examples):
        for learner_ind, learner in enumerate(learners):
            x = train_x[train_index, :].reshape([1, -1])
            y = train_y[train_index].reshape([-1])
            loss, label = learner.learn(x, y)
            loss = np.asscalar(loss)
            label = label.flatten()
            train_acc[learner_ind, train_index] = np.sum(label == y)

            if train_index - window_size >= 0:
                acc = np.mean(train_acc[learner_ind, train_index - window_size: train_index])
                # acc = np.mean(train_acc[method_ind, :train_index])
                logger.info('%s, %dth example, average accuracy %f' %
                            (methods[learner_ind].__name__, train_index, acc))
            else:
                logger.info('%s, %dth example %d' %
                            (methods[learner_ind].__name__, train_index, label == y))

train(0.001)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import utils
import math
import sys
import os
import numpy as np
import copy
import drnn
import rnn_cell_extensions
from tensorflow.python.ops import variable_scope
from sklearn import metrics
from sklearn.cluster import KMeans
from numpy import linalg as LA
import warnings
import train_model
import test_model
import visible_Data
#import train as Train_Model

'''
Tunning Hyperparameter:
hidden_size = [100, 50, 50],[50,30,30]
dilations = [1, 2, 4],[1,4,16]
learning_rate = 1e-4
lamda = [1e-3,1e-2,1e-1,1]
'''

class Config(object):
    """Train config.
    Default:
        hidden_size = [100, 50, 50]
        dilations = [1, 2, 4]
        learning_rate = 1e-4
        cell_type = 'GRU'
        lamda = 1
    """
    batch_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 2, 4]
    num_steps = None
    embedding_size = None
    learning_rate = 1e-4
    cell_type = 'GRU'
    lamda = 1
    class_num = None
    denosing = True  # False
    sample_loss = True  # False


def make_Config(hidden_size,dilations,lamda):
    """
    Set the config of training in this function.

    Input:hidden_size,dilations,cell_type,lamda

    Output: a Config class

    batch_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 2, 4]
    num_steps = None
    embedding_size = None
    learning_rate = 1e-4
    cell_type = 'GRU'
    lamda = 1
    class_num = None
    denosing = True  # False
    sample_loss = True  # False

    """
    config = Config()
    config.hidden_size = hidden_size
    config.dilations = dilations
    #config.cell_type = cell_type
    config.lamda = lamda
    return config

def train(filename,config,path,epoch):
    loss = train_model.run_model(filename, config,path,epoch)
    #loss =[11.670362,10.900277,10.463656,10.635075,10.090756,10.955912,10.916703,10.61992,10.554013,10.220108]
    utils.plot_loss(loss)

def test(filename,config,path):
    ri, nmi, acc, test_hidden_val, km_idx = test_model.run_model(filename, config,path)
    visible_Data.plot(test_hidden_val,km_idx)


'''
hidden_size = [[100, 50, 50],[50,30,30]]
dilations = [[1, 2, 4],[1,4,16]]
learning_rate = 1e-4
lamda = [1e-3,1e-2,1e-1,1]
'''
def main():
    # input your filename
    #filename = './Coffee/Coffee_TRAIN'
    #Use BeetleFly UCR databset
    dataset_type = 'Wine'
    Test = dataset_type + ',test'
    Train = dataset_type + ',train'
    lamda = 1e-1
    hidden_sizes = [100, 50, 50] #try1 = [50,30,30]
    dilations = [1, 2, 4]
    config = make_Config(hidden_sizes,dilations,lamda)
    path = '/Model2/model.ckpt'
    epoch = 30
    print("DataSet_type: "+ dataset_type)
    #Train
    #train(Train,config,path,epoch)

    #Test
    test(Test,config,path)


if __name__ == "__main__":
    main()


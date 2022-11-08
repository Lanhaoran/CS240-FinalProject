# -*- coding: utf-8 -*-

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
import build_model


def run_model(train_data_filename, config, path,epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    train_data, train_label = utils.load_data(train_data_filename)
	
    config.batch_size = train_data.shape[0]
    config.num_steps = train_data.shape[1]
    config.embedding_size = 1

    train_label, num_classes = utils.transfer_labels(train_label)
    config.class_num = num_classes

    print('Label:', np.unique(train_label))
    # Save the loss val every epoch
    loss_Every_epoch = []
    with tf.Session(config=gpu_config) as sess:
        model = build_model.RNN_clustering_model(config=config)
        input_tensors, loss_tensors, real_hidden_abstract, F_update, output_tensor = model.build_model()
        #print(input_tensors.shape, loss_tensors, real_hidden_abstract.shape, F_update.shape, output_tensor.shape)
        train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_tensors['loss'])

        sess.run(tf.global_variables_initializer())

        Epoch = epoch

        for i in range(Epoch):
            # shuffle data and label
            indices = np.random.permutation(train_data.shape[0])
            shuffle_data = train_data[indices]
            shuffle_label = train_label[indices]

            #print(shuffle_data.shape,shuffle_label.shape)
            row = train_data.shape[0]
            batch_len = int(row / config.batch_size)
            left_row = row - batch_len * config.batch_size

            if left_row != 0:
                need_more = config.batch_size - left_row
                rand_idx = np.random.choice(np.arange(batch_len * config.batch_size), size=need_more)
                shuffle_data = np.concatenate((shuffle_data, shuffle_data[rand_idx]), axis=0)
                shuffle_label = np.concatenate((shuffle_label, shuffle_label[rand_idx]), axis=0)
            assert shuffle_data.shape[0] % config.batch_size == 0

            noise_data = np.random.normal(loc=0, scale=0.1, size=[shuffle_data.shape[0]*2, shuffle_data.shape[1]])
            total_abstract = []
            print('----------Epoch %d----------' % i)
            k = 0

            for input, _ in utils.next_batch(config.batch_size, shuffle_data):
                noise = noise_data[k * config.batch_size * 2: (k + 1) * config.batch_size * 2, :]
                fake_input, train_real_fake_labels = utils.get_fake_sample(input)
                #print(config.embedding_size)
                loss_val, abstract, _ = sess.run(
                    [loss_tensors['loss'], real_hidden_abstract, train_op],
                    feed_dict={input_tensors['inputs']: np.concatenate((input, fake_input), axis=0),
                               input_tensors['noise']: noise,
                               input_tensors['real_fake_label']: train_real_fake_labels
                               })
                aa = abstract.shape[0]
                abstract = abstract[:int(aa/2), :]
                #print(abstract.shape)
                print(loss_val)
                loss_Every_epoch.append(loss_val)
                total_abstract.append(abstract)
                k += 1
                if i % 10 == 0 and i != 0:
                    part_hidden_val = np.array(abstract).reshape(-1, np.sum(config.hidden_size) * 2)
                    #print(part_hidden_val.shape)
                    #print(abstract.shape)
                    #print(config.batch_size)
                    #print(train_data.shape)
                    #print(input_tensors, loss_tensors, real_hidden_abstract, F_update, output_tensor)
                    W = part_hidden_val.T
                    U, sigma, VT = np.linalg.svd(W)
                    #print(sigma,VT)
                    sorted_indices = np.argsort(sigma)
                    topk_evecs = VT[sorted_indices[:-num_classes - 1:-1], :]
                    #print(topk_evecs.T)
                    F_new = topk_evecs.T
                    sess.run(F_update, feed_dict={input_tensors['F_new_value']: F_new})
        saver = tf.train.Saver()
        saver.save(sess, path)
        print("Model saved!")
        return loss_Every_epoch


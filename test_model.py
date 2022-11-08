
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
import shutil
import build_model


def run_model(filename, config, model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    assert(model_path != None)
    testing_data, testing_label = utils.load_data(filename)
    testing_label, num_classes = utils.transfer_labels(testing_label)

    config.class_num = num_classes
    config.embedding_size = 1
    config.batch_size = testing_data.shape[0]
    config.num_steps = testing_data.shape[1]

    test_noise_data = np.zeros(shape=testing_data.shape)

    
    with tf.Session(config=gpu_config) as sess:
        model = build_model.RNN_clustering_model(config=config)
        input_tensors, loss_tensors, hidden_abstract, F_update, output_tensor = model.build_model()
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_gragh('Model2\model.ckpt.meta')
        # 加载上次训练的模型结果
        if os.path.exists("/Model2/checkpoint"):
            print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            saver.restore(sess, model_path)
        
        test_total_abstract = sess.run(hidden_abstract,
                    feed_dict={input_tensors['inputs']: testing_data, 
                    input_tensors['noise']: test_noise_data
                    })

        test_hidden_val = np.array(test_total_abstract).reshape(-1, np.sum(config.hidden_size) * 2)
        km = KMeans(n_clusters=num_classes)
        km_idx = km.fit_predict(testing_data)
        ri, nmi, acc = utils.evaluation(prediction=km_idx, label=testing_label)

        #Origin Kmeans result
        #km1_index = km.fit_predict(testing_data)
    
    return ri, nmi, acc, test_hidden_val,testing_label


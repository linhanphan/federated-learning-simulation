# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client example using TensorFlow for Fashion-MNIST image classification."""


from typing import Dict, Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common import Weights
from functools import reduce

from collections import OrderedDict
from copy import deepcopy
#from fastapi_websocket_pubsub import PubSubClient

from datetime import datetime
import os

import fashion_mnist

DEFAULT_SERVER_ADDRESS = "[::]:8080"


conf = {}
conf["batch_size"]    = 64
conf["nb_classes"]     = 10
conf["nb_ephochs"]     = 5
img_rows, img_cols = 28,28
conf["nb_filters"]     = 32
conf["pool_size"]      = 2
conf["kernel_size"]   = 3

conf["samples"] =[1000,5000,20000]
conf["rounds"] = 5
conf["num_of_edges"] = 3


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def aggregate_sedna(weights):
    num_samples = 1000
    total_size = 2000
    old_weight = [np.zeros(np.array(c).shape) for c in weights]
    updates = []
    for inx, row in enumerate(old_weight):
        for c in weights:
            row += (np.array(c[inx]) * num_samples/total_size)
        updates.append(row.tolist())
    weights = deepcopy(updates)
    return updates


def aggregate(list_of_weights):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples = conf["samples"][0]
    num_examples_total = conf["num_of_edges"]*num_examples
    #num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weight] for weight in list_of_weights
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime




def main():
    
    if not os.path.exists("log"):
        os.makedirs("log")
    log_path = r'log/log_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_path)
    os.makedirs(log_path+r'/data')
    
    result_log = 'Simalation Result\n'
    #declare initial variables
    list_of_models = [None] * conf["num_of_edges"]
    weights = [None] * conf["num_of_edges"]
    #acc_dict = {}
    #loss_dict = {}
    #training_history = {}
    num_of_edges = conf["num_of_edges"]
    rounds = conf["rounds"]
    batch_size = conf["batch_size"]
    nb_ephochs = conf["nb_ephochs"]
    #samples = conf["samples"]
    
    config_file = open(log_path+'/config.txt',"w+")
    for k in conf.keys():
        config_file.writelines(k + ' : ' + str(conf[k]) +'\n')
    config_file.close()
    
    dslists = fashion_mnist.load_data_v2(num_of_edges=num_of_edges, nonidd = False, equal = True, ratio = None, seed = 1, log_path=log_path)
    #print(len(dslists))
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #print('y_test shape: ', y_test.shape)
    x_test, y_test = fashion_mnist.shuffle(x_test, y_test, seed=1)
    idx_by_class= []
    test_by_class = []
    #print(y_test[0:10])
    for c in range(0,10):
        sum_of_class = np.where(y_test == c)
        #print(sum_of_class)
        idx_by_class.append(sum_of_class[0].tolist())

    x_test = fashion_mnist.adjust_x_shape(x_test)
    x_test = x_test.astype("float32") / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    for idx in idx_by_class:
        #print(idx)
        sub_x_test = x_test[idx]
        sub_y_test = y_test[idx]
        #print('x_test shape: ', sub_x_test.shape)
        #print('y_test shape: ', sub_y_test.shape)
        #print(sub_y_test[0:10])
        test_by_class.append((sub_x_test,sub_y_test))
    
    init_acc = [None] * num_of_edges
    init_loss = [None] * num_of_edges
    for edge in range(0,num_of_edges):
        xy_train, xy_test = dslists[edge]
        print('start training on edge ',edge)
        result_log += 'start training on edge {}\n'.format(edge)
        model = fashion_mnist.load_model()
        list_of_models[edge] = model
        xy_train, xy_test = dslists[edge]
        x_train, y_train = xy_train
        x_test, y_test = xy_test
        #print('x_train shape: {} \ny_train shape: {}'.format(x_train.shape,y_train.shape))
        result_log += 'x_train shape: {} \ny_train shape: {}\n'.format(x_train.shape,y_train.shape)
        #print('x_test shape: ', x_test.shape)
        #print('y_test shape: ', y_test.shape)
        
        model.fit(x_train,y_train,batch_size= conf["batch_size"], epochs = nb_ephochs, verbose=0)
        
        score = model.evaluate(x_test, y_test, verbose=0)
    
        print('edge {} , loss : {:.5f}, acc : {:.5f}'.format(edge,score[0],score[1]*100))
        result_log += 'edge {} , loss : {:.5f}, acc : {:.5f}\n'.format(edge,score[0],score[1]*100)
        init_loss[edge] = score[0]
        init_acc[edge] = score[1]*100
        weights[edge] = model.get_weights()
        
        if edge == 0:
            for layer in weights[edge]:
                print('shape: {} size: {}'.format(layer.shape,layer.size))
                print('total non-zero',np.count_nonzero(layer))
        class_acc = [None] * 10
        for c in range(0,10):
            score = model.evaluate(test_by_class[c][0], test_by_class[c][1], verbose=0)
            print('edge {} , class {} loss : {:.5f}, acc : {:.5f}'.format(edge,c,score[0],score[1]*100))
            result_log += 'edge {} , class {} loss : {:.5f}, acc : {:.5f}\n'.format(edge,c,score[0],score[1]*100)
            class_acc[c] = score[1]*100
        print('Average accuracy of class: {:.8f}'.format(np.mean(np.array(class_acc))))
        result_log += 'Average accuracy of class: {:.8f}\n'.format(np.mean(np.array(class_acc)))
        #print(len(weights[edge]))
        #for e in weights[edge]:
            #print(e)
        #for layer in weights[edge]:
            #print(layer)
    print('\nAverage accuracy at first train: {:.8f}'.format(np.mean(np.array(init_acc))))
    result_log += '\nAverage accuracy at first train: {:.8f} \n\n'.format(np.mean(np.array(init_acc)))
    for i in range(0,rounds):
        acc = [None] * num_of_edges
        loss = [None] * num_of_edges
        print('===============averaging weight at round {} ==============='.format(i))
        result_log += '===============averaging weight at round {} ===============\n'.format(i)
        average_weights = aggregate(weights)  
        for edge in range(0,num_of_edges):
            threshold = 0.005
            if edge >= 0:
                total_size = 0
                total_change = 0;
                for layer in range(len(average_weights)):
                    diff = np.divide(np.absolute(np.subtract(average_weights[layer],weights[edge][layer])),average_weights[layer])
                    #print('layer: {} non-zero {}/{}'.format(layer,np.count_nonzero(diff),diff.size))
                    bigchange = np.where(diff>threshold)
                    #print('change over {:.4f}: {}/{}'.format(threshold,bigchange[0].size,diff.size))
                    total_size +=diff.size
                    total_change +=bigchange[0].size
                    #if layer == 7:
                        #print(average_weights[layer])
                        #print(weights[edge][layer])
                print('********* edge {}: Total changes {}/{} rate {:.3f} *********'.format(edge,total_change,total_size,(total_change*100.0)/total_size))
            list_of_models[edge].set_weights(average_weights)
            #print('accuracy after just by applying weight')
            score = list_of_models[edge].evaluate(x_test, y_test, verbose=0)
            print('apply weight: edge {} , loss : {:.5f}, acc : {:.5f}'.format(edge,score[0],score[1]*100))
            result_log += 'apply weight: edge {} , loss : {:.5f}, acc : {:.5f}\n'.format(edge,score[0],score[1]*100)
        
            #print('retrain with at edge ' + str(edge))
            list_of_models[edge].fit(x_train,y_train,batch_size= batch_size, epochs = nb_ephochs, verbose=0)
        
            #print('accuracy after train')
            score = list_of_models[edge].evaluate(x_test, y_test, verbose=0)
            print('after re-train: edge {} , loss : {:.5f}, acc : {:.5f}'.format(edge,score[0],score[1]*100))
            result_log += 'after re-train: edge {} , loss : {:.5f}, acc : {:.5f}\n'.format(edge,score[0],score[1]*100)
            weights[edge] = list_of_models[edge].get_weights()
            loss[edge] = score[0]
            acc[edge] = score[1]*100
            class_acc = [None] * 10
            '''
            for c in range(0,10):
                score = model.evaluate(test_by_class[c][0], test_by_class[c][1], verbose=0)
                print('edge {} , class {} loss : {:.5f}, acc : {:.5f}'.format(edge,c,score[0],score[1]*100))
                result_log += 'edge {} , class {} loss : {:.5f}, acc : {:.5f}\n'.format(edge,c,score[0],score[1]*100)
                class_acc[c] = score[1]*100
            '''
        #print('Average accuracy of class: {:.8f}'.format(np.mean(np.array(class_acc))))
        #result_log += 'Average accuracy of class: {:.8f}\n'.format(np.mean(np.array(class_acc)))
        #print('******************************')
        print('*********Average accuracy at round {} : {:.5f}'.format(i,np.mean(np.array(acc))))
        result_log += '**********Average accuracy at round {} : {:.5f}\n\n'.format(i,np.mean(np.array(acc)))
        print('========================================================\n')
    
    log_file = open(log_path +'/result.txt',"w+")
    log_file.write(result_log)
    log_file.close()
        
if __name__ == "__main__":
   main()

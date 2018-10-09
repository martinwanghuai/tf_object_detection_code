'''
Created on 2 Oct 2018

@author: martinwang
'''
from math import exp
from random import seed
from random import random

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list();
    hidden_layer = [{'weights': [random() for i in range(n_inputs +1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def train_network(network, dataset, lr, epoches):
    for epoch in range(epoches):
        n_outputs = len(set(row[-1] for row in dataset))
        sum_error = 0.0
        for row in dataset:
            output = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            list = [(expected(i) - output[i])**2 for i in range(len(expected))]
            sum_error = sum(list)
            backward_propagate_error(network, expected)
            update_weights(network, lr, row)
        print('>epoch=%d, lrate=%.3f, %sum_error=%.3f', epoch, lr, sum_error)

def forward_propagate(network, row):
    for i in range(1,len(network)):
        layer = network[i]
        inputs = row
        for j in range(len(layer)):
            new_inputs = []
            neuron = layer[j]
            activation = sum(neuron['weights'][k] * inputs[k] for k in range(len(inputs) - 1)) + neuron['weights'][-1]
            neuron['output'] = 1.0/ (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs    
        
def backward_propagate_error(network, expected):
    for i in reversed(len(network)):
        errors = list()
        layer = network[i]
        if i == len(network) - 1 :
            for j in range(len(layer)):
                neuron = layer[j]
                errors.add(expected[j] - neuron['output'])
        else:
            error = 0.0
            for j in range(len(layer)):
                for neuron_next in len(network[i + 1]):
                    error += neuron_next['delta'] * neuron_next['weights'][j]
                errors.add(error)
        for j in range(len(errors)):
            neuron = layer[j]
            neuron['delta'] = errors[j]* (1-neuron['output'])*neuron['output']

def update_weights(network, lr, row):
    for i in range(len(network)):
        layer = network[i]
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in layer]
        for neuron in layer:
            for j in len(inputs - 1):
                neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += lr * neuron['delta']                     
            

seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019,1.850220317,0],
           [3.06407232,3.005305973,0],
           [7.627531214,2.759262235,1],
           [5.332441248,2.088626775,1],
           [6.922596716,1.77106367,1],
           [8.675418651,-0.242068655,1],
           [7.673756466,3.508563011,1]] 
n_inputs = len(dataset[0])-1
n_outputs = len(set(row[-1] for row in dataset))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, lr=0.5, epoches=100)
for layer in network:
    print(layer)





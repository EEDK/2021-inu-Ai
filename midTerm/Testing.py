import math
from random import seed

'''
Relu 사용을 위한 내용

import numpy as np

def relu(x):
    return np.maximum(0, x)

'''

# weight summary 과정
def activate(weights, inputs):
    activation = weights[-1]

    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# 순방향 전파, 시그모이드 = s(x) = e(x) / 1 + e(x)
def transferNomal(x):
    return math.exp(x) / (1.0 + math.exp(x))

# 순방향 전파 함수
def forwardPropagate(network, inputs):
    for layer in network:
        updateInput = []
        for net in layer:
            activation = activate(net['weight'], inputs)
            net['output'] = transferNomal(activation)
            updateInput.append(net['output'])
        inputs = updateInput
    return inputs



# Make a prediction with a network
def predict(network, expeceted):
    outputs = forwardPropagate(network, expeceted)
    return outputs.index(max(outputs))


# Test making predictions with the network

seed(1)
dataset = [
    [3.5064385449265267, 2.34547092892632525, 0],
    [4.384621956392097, 3.4530853889904205, 0],
    [4.841442919897487, 4.02507852317520154, 0],
    [3.5985868973088437, 4.1621314217538705, 0],
    [2.887219775424049, 3.31523082529190005, 0],
    [9.79822645535526, 1.1052409596099566, 1],
    [7.8261241795117422, 0.6711054766067182, 1],
    [2.5026163932400305, 5.800780055043912, 1],
    [5.032436157202415, 8.650625621472184, 1],
    [4.095084253434162, 7.69104329159447, 1]
]
network = [
    [
        {'weight': [-0.6860256251179954, 1.6391986279465303, 0.8656386006221861], 'output': 0.9999766221827988, 'delta': -6.399043714642953e-06}
        , {'weight': [-0.5847679382509199, 0.5653278882929079, 0.22257578341987955], 'output': 0.8481933965606284, 'delta': 0.01183959285345359}
    ],
    [
        {'weight': [0.9470961864963364, -0.07314796910860888, -0.9018956759236904], 'output': 0.5413806520720025, 'delta': -0.13441812532928651},
        {'weight': [-1.13210406807162, 0.8265002311167667, 0.7313031253516632], 'output': 0.5357529036080855, 'delta': 0.11546834090803118}
    ]
]

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

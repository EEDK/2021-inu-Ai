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
network = [[{'weight': [-0.7733363439317285, 1.9089771804789595, 0.9296108664875258], 'output': 0.9999960619207737, 'delta': -1.0029080868735003e-06}, {'weight': [-1.3684711949325046, 1.2546828770767593, 0.052300060826971265], 'output': 0.9802979202068065, 'delta': 0.0035468872827052506}], [{'weight': [1.3452136531846226, -0.8814446265867915, -1.0628508376831083], 'output': 0.40650090699840014, 'delta': -0.09807156314114958}, {'weight': [-1.5435892731188314, 1.4766082810773076, 0.9578642550735288], 'output': 0.6706214705100787, 'delta': 0.07275586798127454}]]


for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

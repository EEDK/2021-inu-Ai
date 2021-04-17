from random import random
from random import seed
import math

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


# 네트워크 구성 및 weight 초기화
def MLP2(input, hidden, output):
    net = list()
    h_layer = [{'weight': [random() for i in range(input + 1)]} for i in range(hidden)]
    net.append(h_layer)
    o_layer = [{'weight': [random() for i in range(hidden + 1)]} for i in range(output)]
    net.append(o_layer)

    return net


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


# 에러 체크를 위한 역전파
def transferRevere(x):
    return x * (1.0 - x)


# 각 layer error 계산 및 저장 함수
def layerErrorCheck(network, expeceted):
    length = len(network)
    isFirst = True
    i = length - 1

    while i >= 0:
        layer = network[i]
        errors = list()

        if isFirst:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expeceted[j] - neuron['output'])

            isFirst = False

        else:
            for j in range(len(layer)):
                error = 0
                for neuron in network[i + 1]:
                    error += (neuron['weight'][j] * neuron['delta'])
                errors.append(error)

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transferRevere(neuron['output'])

        i -= 1


# weight update 함수 (Learning rate 자유)
def updateWeight(network, row, learingRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weight'][j] += learingRate * neuron['delta'] * inputs[j]
            neuron['weight'][-1] += learingRate * neuron['delta']


# Epoch (시행횟수)를 입력 중 하나로 받는 전체 training 함수 (시행횟수 자유)
def trainNetwork(network, train, learingRate, Epoch, n_outputs):
    for i in range(Epoch):
        sumError = 0
        for row in train:
            outputs = forwardPropagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1

            sumError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            layerErrorCheck(network, expected)
            updateWeight(network, row, learingRate)

        print('epoch=%d, error=%.3f' % (i, sumError))


# Test training algorithm
seed(1)
network = MLP2(2, 2, 2)


trainNetwork(network, dataset, 0.5, 20, 2)
print(network)

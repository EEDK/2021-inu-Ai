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


# 순방향 전파 s(x) = 1 / 1 + e(-x)
def transferNomal(x):
    return 1.0 / (1.0 + math.exp(-x))


# 순방향 전파 함수
def forwardPropagate(network, inputs):
    for layer in network:
        newInput = []
        for net in layer:
            activation = activate(net['weight'], inputs)
            net['output'] = transferNomal(activation)
            newInput.append(net['output'])
        inputs = newInput
    return inputs


# 에러 체크를 위한 역전파
def transferRevere(x):
    return x * (1.0 - x)

# 각 layer에서의 error 계산 및 저장 함수
def layerErrorCheck(network, expeceted):
    length = len(network)

    i = length - 1
    while i >= 0:
        layer = network[i]
        errors = list()

        i -= 1


if __name__ == '__main__':
    seed(1)
    network = MLP2(2, 2, 2)
    output = forwardPropagate(network, dataset[0])
    print(network)
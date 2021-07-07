from random import random
from random import seed
import math

# dataset -> {input1, input2, expected}
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

    h_layer = [{'weights': [random() for i in range(input + 1)]} for i in range(hidden)]
    net.append(h_layer)
    o_layer = [{'weights': [random() for i in range(hidden + 1)]} for i in range(output)]
    net.append(o_layer)

    return net


# Sigmoid 함수 1 / 1 + exp(-x)
def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def Sigmoid_Diff(x):
    return x * (1.0 - x)


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = Sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# 제출함수 각 weight에 Delta를 구하는 함수
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        U = network[i]

        # 이경우 추가적인 층 수가 없으므로 i는 층수로 구분가능함 -> 1일경우 U2층 0일경우 U1층
        if i == 1:
            # page Week2 36 U2kj가 미치는 영향 기반
            print('U2')
            for j in range(len(U)):
                U[j]['delta'] = -(U[j]['output'] - expected[j]) * Sigmoid_Diff(U[j]['output'])
        else:
            # page Week2 37 U1ji가 미치는 영향 기반
            print('U1')
            for j in range(len(U)):
                error = 0
                # Sigma 구현과정
                preU = network[i + 1]
                for q in range(len(preU)):
                    error += (preU[q]['output'] - expected[q]) * Sigmoid_Diff(preU[q]['output']) * preU[q]['weights'][j]

                U[j]['delta'] = -Sigmoid_Diff(U[j]['output']) * error


#  각 변수를 입력 dataset 과 outputClass 가 변함에 따라 적용될 수 있도록 구현하였는가? - i_num, h_num, o_num 에 따라 변경됨
if __name__ == '__main__':
    # test backpropagation of error
    network = [
        [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
         {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    expected = [0, 1]
    backward_propagate_error(network, expected)
    for layer in network:
        print(layer)

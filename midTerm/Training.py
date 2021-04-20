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

    # weights 네이밍 부여필요성 느낌
    h_layer = [{'weight': [random() for i in range(input + 1)]} for i in range(hidden)]
    net.append(h_layer)
    o_layer = [{'weight': [random() for i in range(hidden + 1)]} for i in range(output)]
    net.append(o_layer)

    return net


# Sigmoid 함수 1 / 1 + exp(-x)
def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def Sigmoid_Diff(x):
    return x * (1.0 - x)

# Relu 함수
def Relu(x):
    if x > 0:
        return x
    else:
        return 0.0

def Relu_diff(x):
    if x > 0:
        return 1.0
    else:
        return x


# Weight Summary 과정 sum(w * i) + bias
def WeightSummation(weights, inputs):
    activation = 0.0
    bias = weights[-1]

    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]

    activation += bias

    return activation


# 순방향 전파 함수
def FrontPropagation(network, row):
    inputs = row

    for layer in network:
        newInputs = []

        for neuron in layer:
            result = WeightSummation(neuron['weight'], inputs)
            # neuron['output'] = Sigmoid(result)
            neuron['output'] = Relu(result)
            newInputs.append(neuron['output'])

        inputs = newInputs

    return newInputs


# each layer error 계산 및 저장 함수
def BackPropagation(network, expected):
    i = len(network) - 1

    # 역전파를 위해 i는 거꾸로 이동
    while i >= 0:
        layer = network[i]
        errors = list()

        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weight'][j] * neuron['gradient'])
                errors.append(error)

        for j in range(len(layer)):
            neuron = layer[j]
            # neuron['gradient'] = errors[j] * Sigmoid_Diff(neuron['output'])
            neuron['gradient'] = errors[j] * Relu_diff(neuron['output'])


        i -= 1


# Weight 업데이트
def UpdateWeight(network, row, learingRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weight'][j] += learingRate * neuron['gradient'] * inputs[j]
            neuron['weight'][-1] += learingRate * neuron['gradient']


# Epoch (시행횟수)를 입력 중 하나로 받는 전체 training 함수 (시행횟수 자유)
def TrainingNetwork(network, data, learingRate, Epoch, outputNum):
    for i in range(Epoch):
        sumError = 0
        for row in data:
            outputs = FrontPropagation(network, row)
            expected = [0 for i in range(outputNum)]
            expected[row[-1]] = 1

            sumError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

            BackPropagation(network, expected)
            UpdateWeight(network, row, learingRate)

        print('시도횟수=%d, error=%f' % (i + 1, sumError))


#  각 변수를 입력 dataset 과 outputClass 가 변함에 따라 적용될 수 있도록 구현하였는가? - i_num, h_num, o_num 에 따라 변경됨
if __name__ == '__main__':
    seed(1)

    i_num = int(input('input  num : '))
    h_num = int(input('hidden num : '))
    o_num = int(input('output num : '))

    network = MLP2(i_num, h_num, o_num)
    TrainingNetwork(network, dataset, learingRate=1.0, Epoch=50, outputNum=o_num)
    print('network = ', network)

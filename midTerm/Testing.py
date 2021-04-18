import math
from random import seed


# Sigmoid 함수 1 / 1 + exp(-x)
def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# Weight Summary 과정 sum(w * i) + bias
def WeightSummation(weights, inputs):
    activation = 0.0
    bias = weights[-1]

    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]

    activation += bias

    return activation


# 순방향 전파 함수
def FrontPropagatation(network, row):
    inputs = row

    for layer in network:
        newInputs = []

        for neuron in layer:
            result = WeightSummation(neuron['weight'], inputs)
            neuron['output'] = Sigmoid(result)
            # neuron['output'] = Relu(result)
            newInputs.append(neuron['output'])

        inputs = newInputs

    return newInputs


# 예상결과 출력
def predict(network, expected):
    outputs = FrontPropagatation(network, expected)
    return outputs.index(max(outputs))


# 결과값

if __name__ == '__main__':
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

    network = [[{'weight': [-1.0148762378123153, 2.618817928454999, 1.0984781137472952], 'output': 0.9999999619329495,
                 'gradient': -3.036838705271522e-09},
                {'weight': [-3.9738688413378807, 2.828278436418336, -0.7216094622453357], 'output': 0.9909508205887168,
                 'gradient': 0.0007993031857783378}], [
                   {'weight': [3.685934348377776, -4.134802030450585, -1.6965403712979577],
                    'output': 0.11154797258336828, 'gradient': -0.01105496432109289},
                   {'weight': [-3.7409552295133306, 4.209038698065864, 1.7141636291371203],
                    'output': 0.8921826210922317, 'gradient': 0.010371254672341035}]]

    for data in dataset:
        prediction = predict(network, data)
        print('Expected=%d, Got=%d' % (data[2], prediction))

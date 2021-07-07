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
def FrontPropagation(network, row):
    inputs = row

    for layer in network:
        newInputs = []

        for neuron in layer:
            result = WeightSummation(neuron['weight'], inputs)
            neuron['output'] = Sigmoid(result)
            newInputs.append(neuron['output'])

        inputs = newInputs

    return newInputs


# 예측값 구하는 함수
def Predict(network, expected):
    outputs = FrontPropagation(network, expected)

    tmp = outputs[0]
    idx = 0

    for i in range(len(outputs)):
        if (outputs[i] > tmp):
            tmp = outputs[i]
            idx = i

    return idx


# 예측값과 정답을 비교하는 함수
def PrintAccuracy(dataset):
    epoch = 0
    correct = 0

    for data in dataset:
        prediction = Predict(network, data)
        print('예상=%f, 정답=%f' % (prediction, data[-1]))
        epoch += 1

        if prediction == data[-1]:
            correct += 1

    accuracy = (correct / epoch) * 100.0
    print('적중률 :%d%%' %(accuracy))


# 결과값

if __name__ == '__main__':
    seed(1)
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

    network = [[{'weight': [-1.758387928421309, 5.148808197074447, 1.7011960084505013], 'output': 0.9999999606228229,
                 'gradient': -3.6175813598748693e-08},
                {'weight': [-0.8692724951022398, -0.7742623654474717, 1.1828123748980865],
                 'output': 0.019085108981376577, 'gradient': -0.005884705440952523}], [
                   {'weight': [3.1723938986430897, 1.6770179526599727, -3.035928564219526],
                    'output': 0.5551133490727379, 'gradient': -0.1370921908809643},
                   {'weight': [-3.178476816307642, -0.5973523269076357, 2.885929066360645],
                    'output': 0.4274854894754119, 'gradient': 0.1401181435292033}]]

    PrintAccuracy(dataset)

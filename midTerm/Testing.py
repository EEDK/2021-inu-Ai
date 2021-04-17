from math import exp


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weight'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Test making predictions with the network
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

network = [[{'weight': [-0.6860256251179954, 1.6391986279465303, 0.8656386006221861], 'output': 0.9999766221827988, 'delta': -6.399043714642953e-06}, {'weight': [-0.5847679382509199, 0.5653278882929079, 0.22257578341987955], 'output': 0.8481933965606284, 'delta': 0.01183959285345359}], [{'weight': [0.9470961864963364, -0.07314796910860888, -0.9018956759236904], 'output': 0.5413806520720025, 'delta': -0.13441812532928651}, {'weight': [-1.13210406807162, 0.8265002311167667, 0.7313031253516632], 'output': 0.5357529036080855, 'delta': 0.11546834090803118}]]

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

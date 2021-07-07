from random import random
from random import seed
import math


def MLP2(input, hidden, output):
    net = list()

    h_layer = [[random() for i in range(input + 1)] for i in range(hidden)]

    net.append(h_layer)

    o_layer = [[random() for i in range(hidden + 1)] for i in range(output)]

    net.append(o_layer)

    return net

nn = MLP2(2, 2, 2)
print(nn)
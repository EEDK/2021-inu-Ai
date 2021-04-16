import torch
import torch.nn as nn

input = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5)
pool = nn.MaxPool2d(2)
out = conv1(input)
out2 = pool(out)
print(out2.size())
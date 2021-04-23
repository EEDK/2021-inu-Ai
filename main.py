import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import exp

torch.manual_seed(0)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
torch.log(F.softmax(z, dim=1))

print(F.log_softmax(z, dim=1))
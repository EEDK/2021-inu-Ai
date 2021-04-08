import torch
import torch.nn as nn

# Ex1 필터사이즈 11 * 11 보폭 4 패딩 0
conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
input = torch.Tensor(1, 1, 227, 227)
out = conv(input)

# Ex2 필터사이즈 7 보폭 2 패딩 0
conv2 = nn.Conv2d(1, 1, 7, stride=2, padding=0)
input2 = torch.Tensor(1, 1, 64, 64)
out2 = conv2(input2)

# Ex3 필터사이즈 5 보폭 1 패딩 2
conv3 = nn.Conv2d(1, 1, 5, stride=1, padding=2)
input3 = torch.Tensor(1, 1, 32, 32)
out3 = conv3(input3)

print(out3.shape)
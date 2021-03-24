import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.FloatTensor([[0,0], [0, 0.5], [0, 1], [0.5, 0], [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]]).to(device)
Y = torch.FloatTensor([[1], [1], [1], [1], [0], [1], [1], [1], [1]]).to(device)

# 은닉층 없을시
# linear = torch.nn.Linear(2,1, bias=True)

# 은닉층 추가
linear1 = torch.nn.Linear(2,2, bias=True)
linear2 = torch.nn.Linear(2,1, bias=True)

sigmoid = torch.nn.Sigmoid()

# model = torch.nn.Sequential(linear, sigmoid).to(device)
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(X)
    cost=criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())


with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach())
    print('\nCorrect: ', predicted.detach())
    print('\nAccuracy: ', accuracy.item())

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
linear = torch.nn.Linear(2,1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid).to(device)
criterion=torch.nn.BCELoss().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=1)

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


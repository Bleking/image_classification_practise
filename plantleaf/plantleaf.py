# 한 줄씩 따라 해보는 파이토치 딥러닝 프로젝트 모음집

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from torchvision import models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

BATCH_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root='./dataset/train/', transform=transform)
valid_dataset = ImageFolder(root='./dataset/validtion/', transform=transform)
test_dataset = ImageFolder(root='./dataset/test/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_worker=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_worker=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_worker=4)

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

# 모델

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.dropout_025 = nn.Dropout(p=0.25)
    self.dropout_05 = nn.Dropout(p=0.25)

    self.fc1 = nn.Linear(4096, 512)
    self.fc2 = nn.Linear(512, 15)  # 클래스 개수가 15
    self.log_softmax =nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.dropout_025(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.dropout_025(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.dropout_025(x)

    x = x.view(-1, 4096)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout_05(x)
    x = self.fc2(x)

    x = self.log_softmax(x)

    return x


model = Net().to(DEVICE)

# 최적화 및 손실 함수 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, dataloader):
  model.eval()
  loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for data, target in dataloader:
      data, target = data.to(DEVICE), target.to(DEVICE)
      output = model(data)

      loss += criterion(output, target).item()

      _, prediction = output.max(1)
      total += target.size(0)
      # correct += (prediction == torch.argmax(target, dim=1)).sum().item()
      correct += prediction.eq(target).sum().item()
  
  loss /= len(dataloader.dataset)
  accuracy = 100.0 * correct / total

  return loss, accuracy


best_acc = 0.0
best_model = copy.deepcopy(model.state_dict())

for epoch in range(1, 30 + 1):
  start = time.time()

  model.train()
  for batch, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

  train_loss, train_acc = evaluate(model, train_loader)
  val_loss, val_acc = evaluate(model, valid_loader)

  if val_acc > best_acc:
    best_acc = val_acc
    best_model = copy.deepcopy(model.state_dict())

  time_elapsed = time.time() - start
  print("\nEPOCH {}".format(epoch))
  print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))   
  print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
  print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

torch.save(model.load_state_dict(best_model), 'plantleaf_cls.pt')

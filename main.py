import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lbcnn import SimpleNetLBC, SimpleNetCNN
import time
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
# build model
model = SimpleNetLBC(num_classes=10).to(device)
# model = SimpleNetCNN(num_classes=10).to(device)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


num_epochs = 1
num_classes = 10
batch_size = 100
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
strat = time.time()
for eopch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            end = time.time()
            print('epoch %d, iter %d: loss %.3f, time: %.3f' % (eopch+1, i+1, loss.item(), (end-strat)))
            strat = time.time()

# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'model.ckpt')
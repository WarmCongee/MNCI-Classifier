import torchvision
import torch
from torchvision import datasets, transforms
from dlwutils.parameters import *
from dlwutils.network import *

# Load
# 将通道范围从0-255变换到0-1之间，然后Normalize
transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

batch_size = 64

train_set = datasets.MNIST(root="./data/", train=True,
                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net = MNISTNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


for epoch in range(20):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, 20))
    print("-" * 10)
    for data in train_loader:
        X_train, y_train = data[0].to(device), data[1].to(device)
        outputs = net(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(running_loss / len(train_set),
                                                                                      100 * running_correct / len(
                                                                                          train_set)))
torch.save(net.state_dict(), MNISTPATH)

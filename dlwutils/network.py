import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 构建卷积函数
        self.conv1 = nn.Conv2d(3, 36, 5)
        # 池化层经验提取函数
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(36, 72, 5)
        # 线性全连接函数
        self.fc1 = nn.Linear(72 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 至此经历了交替两次卷积和池化
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = CIFARNet()

CIFARcriterion = nn.CrossEntropyLoss()
CIFARoptimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 对于不能整除的数值，卷积向下取整，池化向上取整。
        # 构建卷积函数
        self.conv1 = nn.Conv2d(1, 64, 3)
        # 池化层经验提取函数
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        # 线性全连接函数
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 至此经历了交替两次卷积和池化
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
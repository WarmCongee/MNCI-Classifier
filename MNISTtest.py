import torch
import torchvision
import torchvision.transforms as transforms
from dlwutils.show import imshow
from dlwutils.network import MNISTNet
from torch.autograd import Variable
from dlwutils.parameters import *

transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

batch_size = 64

test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



net = MNISTNet()
net.load_state_dict(torch.load(MNISTPATH))

X_test, y_test = next(iter(test_loader))
inputs = Variable(X_test)
pred = net(inputs)
_, pred = torch.max(pred, 1)

print("Predict Label is:", [ i for i in pred.data])
print("Real Label is:",[i for i in y_test])


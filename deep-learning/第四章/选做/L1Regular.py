import torch
import time
import torchvision  # datasets and pretrained neural nets
import torchvision.transforms as transforms

from Regularization import Regularization


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(32 * 32 * 3, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.48216, 0.44653),
                          (0.24703, 0.24349, 0.26159))])
trainset = torchvision.datasets.CIFAR10(root='E:\\KaiKeBa\\基础班\\Python\\第六章\\第四节\\4-CNN(2)\\data',
                                        train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net()
weight_decay=0.1 # 正则化参数
# 初始化正则化
if weight_decay>0:
   reg_loss=Regularization(net, weight_decay, p=1)
else:
   print("no regularization")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
num_epochs = 10

since = time.time()
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.view(-1, 32 * 32 * 3)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        if weight_decay > 0:
            loss = loss + reg_loss(net)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / trainloader.dataset.data.shape[0]
    epoch_acc = running_corrects.double() / trainloader.dataset.data.shape[0]

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

# train Loss: 7.9344 Acc: 0.0986
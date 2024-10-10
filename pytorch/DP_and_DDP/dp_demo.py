import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 假设我们有一个简单的数据集类
class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# 假设我们有一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 10
X = torch.randn(n_sample, n_dim)
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 注意：刚创建的模型是在 cpu 上的 ===== #
device_ids = [0, 1]
model = SimpleModel(n_dim).to(device_ids[0])
model = nn.DataParallel(model, device_ids=device_ids)


optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
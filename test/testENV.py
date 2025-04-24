import torch
import torch.nn as nn
import torch.optim as optim

print(torch.__version__)  # 显示 PyTorch 版本
print(torch.cuda.is_available())  # True，表示 GPU 可用
print(torch.cuda.get_device_name(0))  # 显示你的 RTX 3050

# 创建一个 3x3 矩阵
x = torch.rand(3, 3)
print(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(3, 3).to(device)
print(x)


# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)  # 输入 10 维，输出 1 维

    def forward(self, x):
        return self.fc(x)


# 创建模型
model = SimpleNN()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练步骤
x_train = torch.rand(100, 10)
y_train = torch.rand(100, 1)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

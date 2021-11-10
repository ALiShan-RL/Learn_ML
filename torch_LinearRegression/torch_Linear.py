import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(4) + 0.2 * torch.rand(x.size())                 # 造数据和加噪声

# torch can only train on Variable, so convert them to Variable

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)  # 随机梯度下降
loss_func = torch.nn.MSELoss()  # 使用MSELoss

plt.ion()   # 交互式显示图片

for t in range(300):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # 将梯度置为0，为下次传播做准备
    loss.backward()         # 反向传播，求梯度
    optimizer.step()        # 应用梯度

    if t % 5 == 0:          # 五次一显示

        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())                     # 画数据点
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)   # 画预测曲线
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
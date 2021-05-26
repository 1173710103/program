
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

#读取类别文件
label_csv = pd.read_csv("./facedemo/label.csv", index_col = 'index')

#读取图像文件，并转换格式
trainsets = datasets.ImageFolder("./facedemo",transform = transforms.ToTensor())

#修改label
i = 0
for data in label_csv.values:
    trainsets.imgs[i] = tuple(data)
    i+=1
# 4.定义超参数
BATCH_SIZE = 3 # 每批读取的数据大小
EPOCHS = 100 # 训练10轮

#创建Dataloader
train_loader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)

# 7. 定义RNN模型
class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # （layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out
# 8. 初始化模型
input_dim = 17 # 输入维度
hidden_dim = 100 # 隐层的维度
layer_dim = 2 # 2层RNN
output_dim = 100 # 输出维度

model = RNN_Model(input_dim, hidden_dim, layer_dim, output_dim)
print(model)

# 判断是否有GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 9. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 10. 定义优化器
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 11. 输出模型参数信息
length = len(list(model.parameters()))
# 12. 循环打印模型参数
for i in range(length):
    print('参数：%d'%(i+1))
    print(list(model.parameters())[i].size())
# 13. 模型训练
sequence_dim = 17 # 序列长度
loss_list = [] # 保存loss
accuracy_list = [] # 保存accuracy
iteration_list = [] # 保存循环次数

iter = 0 
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        ii = images[0][0]+images[0][2]
        for b in range(BATCH_SIZE-1):
            ii = torch.cat((ii,images[b+1][0]+images[b+1][2]),0)
        images = ii
        model.train() # 声明训练
        # 一个batch的数据转换为RNN的输入维度
        images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)
        # 梯度清零（否则会不断累加）
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数器自动加1
        iter += 1
        # 模型验证
        if iter % 50 == 0:
            model.eval() # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集，获取数据、预测
            for images, labels in train_loader:
                ii = images[0][0]+images[0][2]
                for b in range(BATCH_SIZE-1):
                    ii = torch.cat((ii,images[b+1][0]+images[b+1][2]),0)
                images = ii
                images = images.view(-1, sequence_dim, input_dim).to(device) 
                # 模型预测
                outputs = model(images)
                # 获取预测概率最大值的下标
                predict = torch.max(outputs.data, 1)[1]
                # 统计测试集的大小
                total += labels.size(0)
                print(total)
                # 统计判断/预测正确的数量
                if torch.cuda.is_available():
                    correct += (predict.gpu() == labels.gpu()).sum()
                else:
                    correct += (predict == labels).sum()
                    print(correct)
            # 计算
            accuracy = correct / total * 100
            print(accuracy)
            # 保存accuracy, loss, iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("loop : {}, Loss : {}, Accuracy : {}".format(iter, loss.item(), accuracy))
            
# 可视化 loss
plt.plot(iteration_list, loss_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Loss')
plt.title('RNN')
plt.show()

# 可视化 accuracy
plt.plot(iteration_list, accuracy_list, color='r')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('RNN')
plt.savefig('LSTM_mnist.png')
plt.show()

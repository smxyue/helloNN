import random
import torch
import torch.nn as nn

import pandas
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

def generate_random(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# 输入参数size必须是整数（integer）类型
def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0,size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor

class Discriminator(nn.Module):

    def __init__(self):
        # 初始化PyTorch父类
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(784+10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.BCELoss()

        # 创建优化器, 使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

        # Load saved model if exists
        try:
            self.load_state_dict(torch.load('discriminator.pth'))
            print("Discriminator: Loaded saved model")
        except FileNotFoundError:
            print("Discriminator: No saved model found")


    def forward(self, image_tensor, label_tensor):
        # 拼接种子和标签
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)
    
    def train(self, inputs, label,targets):
        # 计算网络的输出
        outputs = self.forward(inputs,label)

        # 计算损失值
        loss = self.loss_function(outputs, targets)

        # 每训练10次增加计数器
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        if (self.counter % 10000 == 0):
            print("Discriminator counter = ", self.counter)


        # 归零梯度，反向传播，并更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['Discriminator loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, 
        marker='.', grid=True, yticks=(0, 0.25, 0.5,10,5.0))
        plt.show()


    

class Generator(nn.Module):

    def __init__(self):
        # 初始化PyTorch父类
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(1+10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.BCELoss()

        # 创建优化器，使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

        # Load saved model if exists
        try:
            self.load_state_dict(torch.load('generator.pth'))
            print("Generator: Loaded saved model")
        except FileNotFoundError:
            print("Generator: No saved model found")

    def forward(self, seed_tensor, label):
        # 拼接种子和标签
        inputs = torch.cat((seed_tensor, label))
        return self.model(inputs)
    def train(self, D, inputs,lable,targets):
        # 计算网络输出
        g_output = self.forward(inputs,lable)

        # 输入鉴别器
        d_output = D.forward(g_output,lable)

        # 计算损失值
        loss = D.loss_function(d_output, targets)
        # 每训练10次增加计数器
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        if (self.counter % 10000 == 0):
            print("Generator counter = ", self.counter)

        # 梯度归零，反向传播，并更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()



    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['Generator loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, 
        marker='.', grid=True, yticks=(0, 0.25, 0.5,1.0,5.0))
        plt.show()


def show_generated_image(G):
    # 在3列2行的网格中生成图像
    f, axarr = plt.subplots(4,5, figsize=(16,8))
    
    for i in range(4):
        for j in range(5):
            digital = j+3#random.randint(0,9)
            random_label = torch.zeros(10)
            random_label[digital] = 1.0
            seed=generate_random(1)
            output = G.forward(seed,random_label)
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            axarr[i,j].set_title("{}".format(digital)+":"+str(seed.numpy()))
            axarr[i,j].axis('off')
    plt.show()

def train_gan():
    mnist_dataset = MnistDataset()
    # 创建鉴别器和生成器

    D = Discriminator()
    G = Generator()

    epochs = 1

    for epoch in range(epochs):
        print ("epoch = ", epoch + 1)

        # train Discriminator and Generator

        for label, image_data_tensor, label_tensor in mnist_dataset:
            # 使用真实正样本训练鉴别器
            D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))

            # 为鉴别器生成一个随机独热标签
            random_label = generate_random_one_hot(10)

            # 使用负样本训练鉴别器
            # 使用detach()以避免计算生成器G中的梯度
            D.train(G.forward(generate_random_seed(1), random_label).detach(), random_label,torch.FloatTensor([0.0]))

        
            # 训练生成器
            G.train(D, generate_random_seed(1), label_tensor, torch.FloatTensor([1.0]))
    # Save the trained models
    torch.save(D.state_dict(), 'discriminator.pth')
    torch.save(G.state_dict(), 'generator.pth')
    print("Models saved to disk")
    D.plot_progress()
    G.plot_progress()
    show_generated_image(G)
    # 内存消耗汇总
    #print(torch.cuda.memory_summary(device, abbreviated=True))
def test_discriminator():
    mnist_dataset = MnistDataset()
    D = Discriminator()

    for label, image_data_tensor, target_tensor in mnist_dataset:
        # 真实数据
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        # 生成数据
        D.train(generate_random(784), torch.FloatTensor([0.0]))
        
def test_generator():
    G = Generator()
    output = G.forward(generate_random(1))
    img = output.detach().numpy().reshape(28,28)
    plt.imshow(img, interpolation='none', cmap='Blues',vmin=0, vmax=1)
    plt.show()
def test_gan_variance():
    G=Generator()
    samelist=torch.zeros(10)
    for i in range(10):
        digitals=[]
        lable_tensor=torch.zeros(10)
        lable_tensor[i]=1.0
        for j in range(10000):
            digitals.append(G.forward(generate_random(1),lable_tensor))
        samcount= 0
        for k in range(len(digitals)-1):
            for m in range(k+1,len(digitals)-1):
                if torch.equal(digitals[k], digitals[m]):
                    samcount+=1
        if samcount>0:
            samelist[i]+=samcount
    print(samelist)
    plt.bar(range(10),samelist)
    plt.show()
    pass

if __name__ == "__main__":
    pass
    #test_gan_variance()
    train_gan()
    #G=Generator()
    #show_generated_image(G)
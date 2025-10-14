import os
import torch, random, math, time
from torch import nn

# ---------------- 1. 动态数据生成器 ----------------
class SquareDataset(torch.utils.data.IterableDataset):
    """
    每次迭代都随机产生新的样本，区间边长随训练过程线性放大，
    从而最终覆盖整个实数轴。
    """
    def __init__(self, expand_speed=1.05):
        super().__init__()
        self.expand_speed = expand_speed   # 每 1000 步放大系数
        self.step = 0
    def __iter__(self):
        while True:
            self.step += 1
            L = 5.0 * (self.expand_speed ** (self.step / 1000))  # 动态边界
            x = torch.empty(512, 1).uniform_(-L, L)
            y = x ** 2
            yield x, y

# ---------------- 2. 网络结构 ----------------
net = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
if os.path.exists("kimi_2f.pth"):
    net.load_state_dict(torch.load("kimi_2f.pth"))
    print("Loaded pre-trained weights from kimi_2f.pth")
# ---------------- 3. 训练 ----------------
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
crit = nn.MSELoss()
loader = torch.utils.data.DataLoader(SquareDataset(), batch_size=None)

t0 = time.time()
for i, (x, y) in enumerate(loader):
    opt.zero_grad()
    loss = crit(net(x), y)
    loss.backward()
    opt.step()
    if i % 1000 == 0:
        # 用更大的“测试集”肉眼检查
        with torch.no_grad():
            test = torch.linspace(-10, 10, 1000).unsqueeze(1)
            err = crit(net(test), test**2).item()
            print(f"step {i:5d} | loss={loss.item():.2e} | [-10,10] err={err:.2e}")
    if i > 2000:  # 2 万步足够
        break
print("耗时:", time.time()-t0, "s")
torch.save(net.state_dict(), "kimi_2f.pth")
# ---------------- 4. 随便抽查 ----------------
with torch.no_grad():
    testset=(torch.randint(0,1000,(10,))).float()
    print("随机抽查10个数的平方:")
    for x in testset:
        pred = net(torch.tensor([[x]])).item()
        true = x**2
        print(f"x={x:10.3f} | 预测={pred:12.3f} | 真实={true:12.3f} | 相对误差={abs(pred-true)/true:.2e}")
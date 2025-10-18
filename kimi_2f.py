import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

epochs = 100
MODEL_PATH = 'kimi2f.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def build_model():
    return nn.Sequential(
        nn.Linear(1, 1024), 
        nn.SiLU(), 
        nn.Linear(1024, 1024), 
        nn.SiLU(), 
        nn.Linear(1024, 1)
    )



# 数据生成
def get_data():

    x = torch.linspace(-5000, 5000, 1000).unsqueeze(1)
    y = x ** 2
    return x, y
# 归一化函数
def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x_norm, mean, std):
    return x_norm * std + mean

# 初始化
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    x = checkpoint['x']
    y = checkpoint['y']
    x_mean = checkpoint['x_mean']
    x_std = checkpoint['x_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    model = build_model()
    model.load_state_dict(checkpoint['model_state'])
    print("preTrained Model loaded...")
else:
    x, y = get_data()
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    print("start from scratch...")
    model = build_model()

model = model.to(device)
print(f"x_mean={x_mean:.4f}, x_std={x_std:.4f}, y_mean={y_mean:.4f}, y_std={y_std:.4f}")

x_norm = normalize(x, x_mean, x_std)
y_norm = normalize(y, y_mean, y_std)
x_norm = x_norm.to(device)
y_norm = y_norm.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
def trainMode():
    global x, y, x_norm, y_norm
    total_loss = 0
    print(f"Training(with {device})...")
    starttime = time.time()      
    for i in range(1,2000):
        noise = x_norm * (1.0 + torch.randn(x_norm.size(0),1).to(device)*0.05)
        pred_norm = model(noise)
        loss = loss_fn(pred_norm, y_norm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if i % 100 == 0:  
           print(f'{i:8d}/:Ave Loss: {total_loss/(i):.10f}')
    endtime = time.time()
    print(f'Total time: {endtime-starttime:.2f}s')
    torch.save({
        'x': x,
        'y': y,
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'model_state': model.state_dict()
    }, MODEL_PATH)
    print("trained Model saved...")
    show_result()

def show_result():
    tx = torch.randint(-5000, 5000, (1000, 1), dtype=torch.float32)
    ty = tx ** 2
    tx_norm = normalize(tx, x_mean, x_std).to(device)
    with torch.no_grad():
        pred_norm = model(x_norm)
        pred = denormalize(pred_norm, y_mean, y_std)
        tpred_norm = model(tx_norm)
        tpred = denormalize(tpred_norm, y_mean, y_std)

    plt.scatter(x.cpu().flatten(), y.cpu().flatten(), s=1, label='True 1 y=x²')
    plt.scatter(x.cpu().flatten(), pred.cpu().flatten(), s=1, label='Pred 1')
    plt.scatter(tx.cpu().flatten(), ty.cpu().flatten(), s=1, label='True 2 y=x²')
    plt.scatter(tx.cpu().flatten(), tpred.cpu().flatten(), s=1, label='Pred 2')
    plt.legend()
    plt.show()

def show_result2(start,stop):
    tx = torch.linspace(start, stop, 100).unsqueeze(1)
    ty = tx ** 2
    tx_norm = normalize(tx, x_mean, x_std).to(device)
    with torch.no_grad():
        tpred_norm = model(tx_norm)
        tpred = denormalize(tpred_norm, y_mean, y_std)
    plt.scatter(tx.cpu().flatten(), ty.cpu().flatten(), s=1, label='True 2 y=x²')
    plt.scatter(tx.cpu().flatten(), tpred.cpu().flatten(), s=1, label='Pred 2')
    plt.legend()
    plt.show()

def test2():
    startpoint = torch.randint(0, 4000, (1,)).item()
    mylist=torch.linspace(startpoint, startpoint + 1000, 10)
    fig,axes = plt.subplots(10,1)
    for i in range(len(mylist) -1):
        start=mylist[i]
        end=mylist[i+1]
        val = torch.linspace(start, end, 100).unsqueeze(1)
        val_norm = normalize(val, x_mean, x_std).to(device)
        target_norm = model(val_norm)
        target = denormalize(target_norm.cpu().flatten(), y_mean, y_std)
        axes[i].plot(val.cpu().numpy(), target.cpu().detach().numpy(),label="prediction",color="red")
        axes[i].plot(val.cpu().numpy(), (val**2).cpu().detach().numpy(),label="true",color="blue")
        axes[i].set_title(f"{start:.2f} to {end:.2f}")
        axes[i].legend()
    plt.tight_layout()
    plt.show()
def fx(x_val):
    if isinstance(x_val, (int, float)):
        x_val = torch.tensor([[x_val]], dtype=torch.float32)
    elif isinstance(x_val, torch.Tensor) and x_val.dim() == 0:
        x_val = x_val.unsqueeze(0).unsqueeze(0)
    elif isinstance(x_val, torch.Tensor) and x_val.dim() == 1:
        x_val = x_val.unsqueeze(1)
    x_val_norm = normalize(x_val, x_mean, x_std).to(device)
    with torch.no_grad():
        pred_norm = model(x_val_norm)
        pred = denormalize(pred_norm, y_mean, y_std)
    return pred

def test():
    for i in range(6):
        val = float(input(f"[{6-i}]请输入一个数字:"))
        out = fx(val).item()
        print(f"x={val} :{out} Δ={val**2-out}")

if __name__ == '__main__':
    trainMode()
    show_result2(100,300)
    #test2()
    test()
    pass
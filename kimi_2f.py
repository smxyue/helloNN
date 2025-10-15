import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

epochs = 100
MODEL_PATH = 'kimi2f.pt'

def build_model():
    return nn.Sequential(
        nn.Linear(1, 64), 
        nn.ReLU(),
        nn.Linear(64, 64), 
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# 数据生成
def get_data():

    x = torch.linspace(-5000, 5000, 100000).unsqueeze(1)
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
print(f"x_mean={x_mean:.4f}, x_std={x_std:.4f}, y_mean={y_mean:.4f}, y_std={y_std:.4f}")

x_norm = normalize(x, x_mean, x_std)
y_norm = normalize(y, y_mean, y_std)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
def trainMode():
    global x, y, x_norm, y_norm
    total_loss = 0      
    for i in range(1,2000):
        pred_norm = model(x_norm)
        loss = loss_fn(pred_norm, y_norm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if i % 100 == 0:  
           print(f'{i:4d}/:Ave Loss: {total_loss/(i):.10f}')
    
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
    tx_norm = normalize(tx, x_mean, x_std)
    with torch.no_grad():
        pred_norm = model(x_norm)
        pred = denormalize(pred_norm, y_mean, y_std)
        tpred_norm = model(tx_norm)
        tpred = denormalize(tpred_norm, y_mean, y_std)

    plt.scatter(x.flatten(), y.flatten(), s=1, label='True 1 y=x²')
    plt.scatter(x.flatten(), pred.flatten(), s=1, label='Pred 1')
    plt.scatter(tx.flatten(), ty.flatten(), s=1, label='True 2 y=x²')
    plt.scatter(tx.flatten(), tpred.flatten(), s=1, label='Pred 2')
    plt.legend()
    plt.show()

def show_result2():
    tx = torch.linspace(100, 300, 1000).unsqueeze(1)
    ty = tx ** 2
    tx_norm = normalize(tx, x_mean, x_std)
    with torch.no_grad():
        tpred_norm = model(tx_norm)
        tpred = denormalize(tpred_norm, y_mean, y_std)
    plt.scatter(tx.flatten(), ty.flatten(), s=1, label='True 2 y=x²')
    plt.scatter(tx.flatten(), tpred.flatten(), s=1, label='Pred 2')
    plt.legend()
    plt.show()

def fx(x_val):
    if isinstance(x_val, (int, float)):
        x_val = torch.tensor([[x_val]], dtype=torch.float32)
    elif isinstance(x_val, torch.Tensor) and x_val.dim() == 0:
        x_val = x_val.unsqueeze(0).unsqueeze(0)
    elif isinstance(x_val, torch.Tensor) and x_val.dim() == 1:
        x_val = x_val.unsqueeze(1)
    x_val_norm = normalize(x_val, x_mean, x_std)
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
    show_result2()
    #test()
    pass
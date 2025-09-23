# mnist_lstm.py
import os
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from mylib import create_drawing_window, load_data, mnist_image, plot_image

# 0. 超参数
SEQ_LEN = 28          # 28 行
INPUT_SIZE = 28       # 每行 28 像素
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据：灰度值归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),               # -> [0,1]
    transforms.Lambda(lambda x: x.view(SEQ_LEN, INPUT_SIZE))  # 拉成 28×28
])
train_data, val_data, test_data = load_data()

# 将numpy数组转换为PyTorch数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.images = data[0]  # 图像数据
        self.labels = data[1]  # 标签数据
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # data[0][idx]为image, data[1][idx]为label
        image = self.images[idx]
        label = self.labels[idx]
        
        # 如果图像是numpy数组，需要转换为PIL图像以便使用transforms
        if isinstance(image, np.ndarray):
             # 假设原始图像是28x28，但被展平成784维向量
            if image.ndim == 1 and image.shape[0] == 784:
                image = image.reshape(28, 28)
            # 转换为浮点型并归一化到[0,1]范围
            image = image.astype(np.float32)
            # 转换为PIL图像 (需要添加通道维度)
            image = transforms.ToPILImage()(image)
            
        if self.transform:
            image = self.transform(image)
        return image, label

train_set = CustomDataset(train_data, transform=transform)
test_set = CustomDataset(test_data, transform=transform)
# 如果需要验证集，也可以创建 val_set = CustomDataset(val_data, transform=transform)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  BATCH_SIZE, shuffle=False, num_workers=0)

# 2. 模型：一层 LSTM + 全连接
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):           # x: [B, 28, 28]
        out, (h_n, _) = self.lstm(x)        # out:[B, 28, H], h_n:[1, B, H]
        last_hidden = h_n[-1]               # [B, H]
        return self.fc(last_hidden)         # [B, 10]

model = LSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model_path = 'mnist_lstm.pt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f'Loaded saved model from {model_path}')
else:
    print('No saved model found. Training from scratch.')
# 3. 训练与测试
def run_one_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(1)
            total   += y.size(0)
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * y.size(0)
    return correct/total, loss_sum/total

def train_mode(model):
    for epoch in range(1, 10):
        tr_acc, tr_loss = run_one_epoch(train_loader, training=True)
        te_acc, te_loss = run_one_epoch(test_loader, training=False)
        print(f'Epoch {epoch}: train acc={tr_acc:.4f}  test acc={te_acc:.4f}')

    # 4. 保存模型（可选）
    torch.save(model.state_dict(), model_path)
    print('Done. Model saved to {}'.format(model_path))

def predict(image):
    
    # Ensure the model is in evaluation mode
    model.eval()
    
       # Preprocess the image
    if isinstance(image, np.ndarray):
        # 如果是numpy数组，检查是否需要reshape
        if image.ndim == 1 and image.shape[0] == 784:
            image = image.reshape(28, 28)
        # 转换为PIL图像
        image = transforms.ToPILImage()(image)
      
    image = transform(image).unsqueeze(0).to(DEVICE)  # 添加批次维度并移到设备
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        predicted_digit = output.argmax(1).item()
    return predicted_digit

def test_draw_predict():
    for i in range(10):
        fn=f'test{i}.png'
        img =mnist_image(fn)
        pred=predict(img)
        print(f'Predict {fn}: {pred}')
def test_on_dataset():
    _,_,test_data = load_data()
    startindex=np.random.randint(0, len(test_data[0])-100)
    binggo=0
    samples=0
    for i in range(100):
        img=test_data[0][startindex+i].reshape(28,28)
        label=test_data[1][startindex+i]
        if label == label:
            samples+=1
            pred=predict(img)
            mark="x"
            if label==pred:
                binggo+=1
                mark=" "
            print(f'Label={label}, Predict={pred} {mark}')
    print(f'Accuracy: {binggo}/{samples} = {binggo/samples:.4f}')
def test_hand_draw():
    for _ in range(10):
        create_drawing_window()
        img =mnist_image("saved.png")
        pred=predict(img)
        print(f'Predict: {pred}')
if __name__ == '__main__':
    #train_mode(model)
    test_draw_predict()
    #test_on_dataset()
    #test_hand_draw()
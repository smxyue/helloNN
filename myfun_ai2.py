from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

# Define device alignment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="F2usenor2.pth"
def normalize(x,scope):
    return (x / scope)

def denormalize(x_norm, scope):
    return (x_norm * scope)

def my_data():
    #x = torch.linspace(-5000, 5000, 10000).tolist()
    x=[0]
    #x.extend(torch.randint(-5000, 5000, (50000,)).tolist())
    for i in range(1,5000):
        for j in range(10):
            x.append(i.__float__()+j*0.1)
            x.append(-(i.__float__()+j*0.1))    
    x = np.array(x)
    np.random.shuffle(x)
    y = x ** 2


    x_norm = normalize(x, 5000)
    y_norm = normalize(y, 25000000)
    

    # 转 Tensor
    X = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)

    return X, Y
def get_data_loader(batch_size=64):
    """Create a data loader from the data"""
    X, Y = my_data()
    dataset = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 2048),
            nn.SiLU(),
            nn.Linear(2048, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=DEVICE)
            self.load_state_dict(state)
            print("Loaded pre-trained model from", model_path)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            print("No pre-trained model found. Initializing with random weights.")
        
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input")
        return self.fc(x)
        pass
        
    

    def train_model(self, epochs=100, learning_rate=0.00001, batch_size=64):
        self.to(DEVICE)

        criterion = nn.MSELoss()  # or appropriate loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        # 使用 DataLoader 的 batch 训练，避免用整个 x 每次更新
        data_loader = get_data_loader(batch_size=batch_size)

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                output = self(data)
                #loss = criterion(output, target)
                loss = torch.sum((output - target)**2)
                loss.backward()

                # 防止梯度爆炸导致损失跳回
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch}/{epochs}: Loss: {avg_loss:.10f}')
            # Step the scheduler based on validation loss
            scheduler.step(avg_loss)

        # 保存训练好的参数
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        # ...existing code...

    
    def predict(self, input_data):
        """
        Use the model for prediction
        """
        self.eval()
        input_data = normalize(input_data, 5000)
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1).to(DEVICE)
        with torch.no_grad():
            output = self(input_data)
            output = denormalize(output, 25000000) 
            return output.item()
        
    def plot_train_loss(self):
        self.eval()
        x, y = my_data()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        with torch.no_grad():
            predictions = self(x)
        
        
        x_np = x.cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        pred_np = predictions.cpu().numpy().flatten()
        
        # Sample a subset for clearer visualization
        indices = np.linspace(0, len(x_np)-1, 1000, dtype=int)
        x_sample = x_np[indices]
        y_sample = y_np[indices]
        pred_sample = pred_np[indices]
        
        
        plt.scatter(x_np, y_np, label='Expect', s=1)
        plt.scatter(x_np, pred_np, label='Network',s=1)
        plt.legend()
        plt.show()
    def test_data(self):
        x_norm,y_norm = my_data()
        x=denormalize(x_norm,5000)
        y=denormalize(y_norm,25000000)
        errors=0
        print("Start testing data...")
        for i in range(len(x)):
            if (x[i]**2 - y[i]) > 3.99 and x[i] > 0.01:
                errors+=1
                print(f"norm({x_norm[i].item()},{y_norm[i].item()}) Real({x[i].item()}, {y[i].item()}) 偏差:{x[i].item()**2 - y[i].item()}")
        print(f"Total data errors: {errors}")
    def test(self):
        for i in range(6):
            val = float(input(f"[{6-i}]请输入一个数字:"))
            with torch.no_grad():
                val_norm = normalize(val, 5000)
                val_tensor = torch.tensor([[val_norm]], dtype=torch.float32).to(DEVICE)
                pred_norm = model(val_tensor)
                pred = denormalize(pred_norm.cpu(), 25000000)
                out = val**2
                print(f"x={val} :{pred.item()} Δ={pred.item()-out}")
if __name__ == "__main__":
    model = MyModel().to(DEVICE)
    #model.train_model()
    #model.plot_train_loss()
    
    test_values=[0,1,-1,-5,-5,10,50,100,500,1000,4000,5000]
    
    
    val_all = 0
    for val in test_values:
        # Apply the same log transform as used in training
        prediction = model.predict(val)
        expected = val ** 2
        print(f"x: {val:12.2f},\t Predicted: {prediction:12.2f},\t  error: {abs(prediction - expected):.2f}")
        val_all += abs(prediction - expected)
    print(f"Sum error: {val_all:.2f}")

    #model.test_data()
    model.test()
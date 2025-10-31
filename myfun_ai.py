from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

# Define device alignment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="F2usenor.pth"
def normalize(x,scope):
    return (x / scope)

def denormalize(x_norm, scope):
    return (x_norm * scope)

def my_data():
    x = torch.linspace(-500, 500, 10000).tolist()
   
    x = np.array(x)
    y = x ** 2


    x_norm = normalize(x, 500)
    y_norm = normalize(y, 250000)
    

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
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
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
        
    

    def train_model(self, epochs=30, learning_rate=0.0001, batch_size=64):
        self.to(DEVICE)

        criterion = nn.MSELoss()  # or appropriate loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

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
                loss = criterion(output, target)
                loss.backward()

                # 防止梯度爆炸导致损失跳回
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch}/{epochs}: Loss: {avg_loss:.10f}')

        # 保存训练好的参数
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        # ...existing code...

    
    def predict(self, input_data):
        """
        Use the model for prediction
        """
        self.eval()
        input_data = normalize(input_data, 500)
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1).to(DEVICE)
        with torch.no_grad():
            output = self(input_data)
            output = denormalize(output, 250000) 
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
    

if __name__ == "__main__":
    model = MyModel().to(DEVICE)
    model.train_model()
    test_values = torch.linspace(0, 500, 20).tolist()
    #test_values+=[-500,500]
    #model.plot_train_loss()
    val_all = 0
    for val in test_values:
        # Apply the same log transform as used in training
        prediction = model.predict(val)
        expected = val ** 2
        print(f"x: {val:12.2f},\t Predicted: {prediction:12.2f},\t  error: {abs(prediction - expected):.2f}")
        val_all += abs(prediction - expected)
    print(f"Sum error: {val_all:.2f}")
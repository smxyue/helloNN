import numpy as np
import torch
import torch.nn as nn
import os

# Define device alignment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="F2uselog.pth"
def my_data():
    x = torch.randint(-5000, 5000, (100000,)).tolist()
    
    x=np.array(x)
    y = x ** 2
    # 对数变换
    x_abs = np.abs(x) + 1e-8  # Slightly larger epsilon
    y_clipped = np.clip(y, 1e-8, None)  # Clip to prevent log(0)
    
    x_log = np.log10(np.abs(x_abs))
    y_log = np.log10(y_clipped)

    # 转 Tensor
    X = torch.tensor(x_log, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(y_log, dtype=torch.float32).view(-1, 1)

    # Check for NaN or Inf values
    if torch.isnan(X).any() or torch.isinf(X).any() or torch.isnan(Y).any() or torch.isinf(Y).any():
        print("Warning: NaN or Inf values detected in data")

    return X, Y
def get_data_loader(batch_size=64):
    """Create a data loader from the data"""
    X, Y = my_data()
    dataset = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
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
        
    
    def train_model(self, epochs=20000, learning_rate=0.0001,batch_size=100):
        self.to(DEVICE)

        criterion = nn.MSELoss()  # or appropriate loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        x,y=my_data()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            #for batch_idx, (data, target) in enumerate(data_loader):
            #    data, target = data.to(DEVICE), target.to(DEVICE)
            x=x.to(DEVICE)
            y=y.to(DEVICE)   
            optimizer.zero_grad()
            output = self(x)
            loss = torch.sum((output-y)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                #if batch_idx % 1000 == 0:
                    #print(f'Epoch {epoch:4d}, Batch {batch_idx:4d}, Loss: {total_loss/((batch_idx+1) * batch_size):.10f}')
            print(f'Epochs{epoch:5d}/{epochs:5d}:Loss: {loss.item():.10f}')
        
        # Save model after training
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def predict(self, input_data):
        """
        Use the model for prediction
        """
        self.eval()
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1).to(DEVICE)
        with torch.no_grad():
            output = self(input_data)
            output = 10 ** output - 1e-6
            return output.item()
    def test_int(self):
        val = [0,-1,1,-5,5,10,50,100,500,1000,4000,5000]
        for v in val:
            log_v = np.log10(np.abs(v) + 1e-6)
            pred = self.predict(log_v)
            expected = v ** 2
            print(f"x: {v}, Predicted: {pred:.2f}, Expected: {expected:.2f}, Error: {abs(pred - expected):.2f}")
if __name__ == "__main__":
    model = MyModel().to(DEVICE)
    model.train_model()
    model.test_int()
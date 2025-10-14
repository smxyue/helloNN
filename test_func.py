import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def fn(x):
    return 8.323*x+23.427

def generate_data(num_samples=10000):
    x = (torch.rand(num_samples, 1) - 0.5) * 1 # [-5000, 5000]
    y = fn(x)
    return TensorDataset(x, y)

dataset = generate_data()
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
class my_func(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        if os.path.exists("my_func2.pth"):
            self.load_state_dict(torch.load("my_func2.pth"))
            print("my_func: Loaded pre-trained weights")

    def forward(self, x):
        return self.model(x)

    def train_model(self):
        for epoch in range(10):  # Example: 5 epochs
            total_loss = 0
            for inputs, targets in data_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")
        
        #torch.save(self.state_dict(), "my_func2.pth")    
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def show_result(self):
        import matplotlib.pyplot as plt
        x = torch.linspace(-5000, 5000, 100).view(-1, 1)
        y = self.predict(x)
        plt.plot(x.numpy(), y.numpy(), label='Predicted',linestyle='dashdot', color='red')
        plt.plot(x.numpy(), fn(x.numpy()), label='True', linestyle='solid', color='blue')
        plt.legend()
        plt.show()  
    def show_result2(self):
        x = torch.linspace(-5000, 5000, 10).view(-1, 1)
        y = self.predict(x)
        for i in range(10):
            true_val = fn(x[i].item())
            pred_val = y[i].item()
            error = abs(pred_val - true_val)
            print(f"Input: {x[i].item():.2f}, Predicted: {pred_val:.2f}, True: {true_val:.2f}, Error: {error:.2f}")

if __name__ == "__main__":
    net = my_func().to('cpu')
    net.train_model()
    net.show_result()
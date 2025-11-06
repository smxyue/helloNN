from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="F2usenor2.pth"
mynet_path="myfun_net.pth"

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
    x1 = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y1= torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, Y,x1,y1
def get_data_loader(batch_size=64):
    """Create a data loader from the data"""
    X, Y,x1,y1= my_data()
    dataset = torch.utils.data.TensorDataset(X, Y,x1,y1)
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
            print("No pre-trained model found. please train a model (using myfun_ai2.py) first.")
        
    def forward(self, x):
        return self.fc(x)
        pass
        
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        if os.path.exists(mynet_path):
            state = torch.load(mynet_path, map_location=DEVICE)
            self.load_state_dict(state)
            print("Loaded pre-trained model from", mynet_path)
        else:
            print("No pre-trained model found.")
        

        
    def forward(self, x):
        return self.fc(x)   
    
    def train_model(self, epochs=10, learning_rate=0.1, batch_size=100):
        self.to(DEVICE)

        data_loader = get_data_loader(batch_size=batch_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.01, patience=3)

        # load base model and freeze it
        base_model = MyModel().to(DEVICE)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

        for epoch in range(1, epochs + 1):
            epoch_loss_sum = 0.0
            n_samples = 0
            self.train()
            for inputs, targets,x1,y1 in data_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                x1 = x1.to(DEVICE, non_blocking=True)
                y1 = y1.to(DEVICE, non_blocking=True)

                with torch.no_grad():
                    base_pred = base_model(inputs)   # base prediction (frozen)
                bp_delta = y1-denormalize(base_pred,25000000.0)    #模型预测值
                optimizer.zero_grad()
                pred_residual = self(inputs)        # network predicts residual
                loss = criterion(pred_residual, bp_delta)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_actual = inputs.size(0)
                epoch_loss_sum += loss.item() * batch_size_actual
                n_samples += batch_size_actual

            avg_loss = epoch_loss_sum / max(1, n_samples)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6e}")
            scheduler.step(avg_loss)

        torch.save(self.state_dict(), mynet_path)
        print("Model saved to", mynet_path)

    def show_result(self):
        base_model = MyModel().to(DEVICE)
        base_model.eval()
        self.to(DEVICE)
        self.eval()

        vals_original = [-1,-0.5,0,0.5,1,2,5,10,50,100,500,1000,3000,5000]
        vals_normalized = normalize(np.array(vals_original), 5000)
        vals_tensor = torch.tensor(vals_normalized, dtype=torch.float32).view(-1, 1).to(DEVICE)

        with torch.no_grad():
            base_out = base_model(vals_tensor)      # normalized
            net_out = self(vals_tensor)             # normalized residual
            combined = base_out + net_out

        base_np = (base_out.cpu().numpy().flatten() * 25000000.0)
        net_np = (net_out.cpu().numpy().flatten() )
        comb_np = base_np + net_np

        for i, x in enumerate(vals_original):
            actual = x ** 2
            print(f"x: {x}, base: {base_np[i]:.2f}, net: {net_np[i]:.2f}, combined: {comb_np[i]:.2f}, actual: {actual:.2f} error: {abs(comb_np[i]-actual):.2f}")
    def test_linespace(self):
        base_model = MyModel().to(DEVICE)
        base_model.eval()
        self.to(DEVICE)
        self.eval()

        x = np.linspace(-5000, 5000, 10)
        x_norm = x/5000.0
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(DEVICE)

        with torch.no_grad():
            base_out = base_model(x_tensor)
            net_out = self(x_tensor)
            base_out = base_out.cpu().numpy().flatten() * 25000000.0
            base_out += net_out.cpu().numpy().flatten()
            for i in range(len(x)):
                actual = x[i] ** 2
                print(f"x: {x[i]}, out: {base_out[i]:.2f}, actual: {actual:.2f}, error: {abs(base_out[i]-actual):.2f}")

if __name__ == "__main__":
    net = MyNet()
    net.train_model()
    net.show_result()
    net.test_linespace()
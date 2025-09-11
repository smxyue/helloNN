import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mylib import create_drawing_window, mnist_image



CHKPOINT_FILE = "basic_conv.pkl"  # Global checkpoint file name



class BasicConvNet(nn.Module):
    def __init__(self):
        super(BasicConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(20 * 12 * 12, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv(x)))
        x = x.view(-1, 20 * 12 * 12)
        x = F.sigmoid(self.fc(x))
        x = self.out(x)
        return x

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_torch_conv_net(CHKPOINT_FILE):
    model = BasicConvNet()
    if os.path.exists(CHKPOINT_FILE):
        print(f"Loading checkpoint from {CHKPOINT_FILE}")
        checkpoint = torch.load(CHKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded with accuracy: {checkpoint.get('accuracy', 'N/A'):.2%}")
        return model
    else:
        print("No checkpoint found, returning untrained model")
        return None
def predict_with_model():
    model=load_torch_conv_net(CHKPOINT_FILE)
    if model is None:
         return None
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = []
    fl=["test0.png","test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.png"]
    for f in fl:
        images.append(mnist_image(f))
    with torch.no_grad():
        for img in images:
            img = torch.tensor(img,dtype=torch.float32).reshape(-1, 1, 28, 28)
            img = img.to(device)
            output = model(img)
            output = F.softmax(output, dim=1)
            max_prob, max_idx = torch.max(output, dim=1)
            print(f"Predicted class: {max_idx.item()} with confidence {max_prob.item():.2%}" )
            ###predicted_class = output.argmax(dim=1)
            ###predicted_confidence = torch.softmax(output, dim=1).max().item()
            ###print(f"Predicted class: {predicted_class.item()} with confidence {predicted_confidence:.2%}")

def predict_with_conv(model,img):
    if model is None:
         return None
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = torch.tensor(img,dtype=torch.float32).reshape(-1, 1, 28, 28)
    img = img.to(device)
    output = model(img)
    output = F.softmax(output, dim=1)
    max_prob, max_idx = torch.max(output, dim=1)
    return max_idx.item(), max_prob.item()
    #print(f"Predicted class: {max_idx.item()} with confidence {max_prob.item():.2%}" )
    ###predicted_class = output.argmax(dim=1)
    ###predicted_confidence = torch.softmax(output, dim=1).max().item()
    ###print(f"Predicted class: {predicted_class.item()} with confidence {predicted_confidence:.2%}")

def draw_test_conv():
    model = load_torch_conv_net(CHKPOINT_FILE)
    if model is None:
         print("No trained model found, returning")
         return None
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(10):
        fn = create_drawing_window()
        image = mnist_image(fn)
        digital,prob =predict_with_conv(model,image)
        print(f"{i}:\t {digital} \t {prob:.2%}")

def get_pytorch_loaders(filename="data/mnist.pkl.gz", batch_size=10):
    import gzip, pickle
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    # 转为 torch.Tensor
    train_x = torch.tensor(training_data[0], dtype=torch.float32).reshape(-1, 1, 28, 28)
    train_y = torch.tensor(training_data[1], dtype=torch.long)
    test_x = torch.tensor(test_data[0], dtype=torch.float32).reshape(-1, 1, 28, 28)
    test_y = torch.tensor(test_data[1], dtype=torch.long)
    # 构建 DataLoader
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def basic_conv1(epochs=10, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(CHKPOINT_FILE):
        print(f"Loading checkpoint from {CHKPOINT_FILE}")
        checkpoint = torch.load(CHKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No checkpoint found, starting training from scratch")



    train_loader, test_loader = get_pytorch_loaders()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished.")

    # 测试准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy= correct / total
    print(f"Test accuracy: {correct / total:.2%}")
            # Save checkpoint
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }, CHKPOINT_FILE)
    print(f"Checkpoint saved to {CHKPOINT_FILE}")

    return model

if __name__ == "__main__":
    basic_conv1()
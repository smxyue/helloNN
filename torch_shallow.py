import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gzip
import pickle
import os

from mylib import create_drawing_window, mnist_image

# Define the neural network with 100 hidden neurons
class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        # Input layer to hidden layer (28*28 -> 100)
        self.fc1 = nn.Linear(28*28, 100)
        # Hidden layer to output layer (100 -> 10)
        self.fc2 = nn.Linear(100, 10)
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28*28)
        # Pass through hidden layer with activation
        x = self.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x

def load_data_shared(filename="data/mnist.pkl.gz"):
    """
    Load the MNIST data from local file and convert to PyTorch tensors.
    This function mimics the behavior of load_data_shared from network3.py
    but returns PyTorch tensors instead of Theano shared variables.
    """
    # Open and load the pickle file
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    
    # Convert to PyTorch tensors
    def convert_to_tensor(data):
        # Convert data to torch tensors
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.long)
        return x, y
    
    # Return data in the same structure as the original function
    return [convert_to_tensor(training_data), 
            convert_to_tensor(validation_data), 
            convert_to_tensor(test_data)]

# Custom Dataset class to work with the loaded data
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        # Get image and label
        image = self.data[0][idx].reshape(28, 28)  # Reshape to 28x28
        label = self.data[1][idx]
        
        # Apply transform if provided
        if self.transform:
            # Add channel dimension for transforms
            image = image.unsqueeze(0)  # Add channel dimension
            image = self.transform(image)
        
        return image, label

def get_train_loader(batch_size=64):
    """
    Create a DataLoader with the training data.
    This replaces the previous load_data_shared function for training purposes.
    """
    # Load data using the updated load_data_shared function
    training_data, validation_data, test_data = load_data_shared()
    
    # Create dataset with transforms
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = MNISTDataset(training_data, transform=transform)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,  # Enable multiprocessing
        pin_memory=True  # Enable memory pinning for faster GPU transfer
    )
    
    return train_loader
# Training function
def train_shallow_net(model):
    # Define transformations for the training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset

     # Load data using the load_data_shared function
    train_loader = get_train_loader()
    print(len(train_loader))
    # Initialize the network
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    
    print('Training completed')
    return model

# Function to predict digits from test images
def predict_digits(model, transform):
    

    model.eval()
    predictions = []
    
    # Load and predict for test0.png to test9.png
    for i in range(10):
        try:
            # Load image
            image_path = f'test{i}.png'
            image = mnist_image(image_path)
            
            
            image_tensor =torch.tensor(image,dtype=torch.float32).reshape(-1, 1, 28, 28)
            
            ###image_tensor = transform(image)
            ###image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)
                predictions.append(predicted.item())
                print(f'test{i}.png: predicted digit is {predicted.item()}')
        except FileNotFoundError:
            print(f'test{i}.png not found')
            predictions.append(None)
    
    return predictions

def predict_data(model, image):
    

    model.eval()
    predictions = []
    
    # Load and predict for test0.png to test9.png
    image_tensor =torch.tensor(image,dtype=torch.float32).reshape(-1, 1, 28, 28)
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    
    
    return predicted.item()
def evaluate_model(model):
        # Load test data
        training_data, validation_data, test_data = load_data_shared()
        
        # Create test dataset and dataloader
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = MNISTDataset(test_data, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(outputs)
                print("********************")
                print(predicted)
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy

def random_test_samples(model, num_samples=10):
    """
    Randomly select samples from test dataset and evaluate model predictions.
    
    Args:
        model: Trained model
        num_samples: Number of random samples to test (default: 10)
    """
    # Load test data
    training_data, validation_data, test_data = load_data_shared()
    
    # Set model to evaluation mode
    model.eval()
    
    # Get random indices
    total_samples = len(test_data[0])
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print(f"Randomly selected {num_samples} test samples:")
    print("-" * 50)
    
    correct_count = 0
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            # Get sample data
            image_flat = test_data[0][idx]
            true_label = test_data[1][idx]
            
            # Reshape and transform image
            image = image_flat.reshape(1, 28, 28)  # Add channel dimension
            image = torch.tensor(image, dtype=torch.float32)
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            output = model(image)
            
            # Get predicted class
            _, predicted = torch.max(output.data, 1)
            predicted_label = predicted.item()
            
            # Get confidence (probability) - apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            confidence = torch.max(probabilities).item()
            
            # Check if prediction is correct
            is_correct = predicted_label == true_label
            if is_correct:
                correct_count += 1
            
            # Display results
            status = "✓" if is_correct else "✗"
            print(f"Sample {i+1:2d}: True={true_label} | Predicted={predicted_label} "
                  f"| Confidence={confidence:.4f} {status}")
    
    accuracy = correct_count / num_samples * 100
    print("-" * 50)
    print(f"Accuracy on {num_samples} random samples: {accuracy:.1f}% ({correct_count}/{num_samples})")
    
    return correct_count, num_samples

def load_my_shallow_net():
    model = ShallowNet()
    model_path = 'shallow_net.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No previously trained model found")
        return None
    return model

def tranin_and_use():
        # Define the same transform used for training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    model = ShallowNet()
    
    # 1. Load previously trained model if it exists
    model_path = 'shallow_net.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded previously trained model")
    else:
        print("No previously trained model found")
    
    # Train the model
    model = train_shallow_net(model)
    
    # Save the model
    torch.save(model.state_dict(), 'shallow_net.pth')
    
    test_accuracy = evaluate_model(model)
    
    # Predict digits in test images
    predictions = predict_digits(model, transform)
    
    
    #correct_count, total_samples = random_test_samples(model, 100)

# Main execution


def test_hand_draw():
    fn = create_drawing_window()
    image = mnist_image(fn)
    model =load_my_shallow_net()
    if (model == None):
        print("Failed to load model")
    else:
        # Call the function with the loaded model and
        got = predict_data(model, image)
        print("Predicted digit:", got)

def test_drawed_digits_shallow():
    model =load_my_shallow_net()
    if (model == None):
        print("Failed to load model")
        return
    
    for i in range(10):
        fn=f'test{i}.png'
        img =mnist_image(fn)
        
        if (model == None):
            print("Failed to load model")
        else:
            # Call the function with the loaded model and
            got = predict_data(model, img)
            print(f'Predict {fn}: {got}')

if __name__ == "__main__":
    #tranin_and_use()
    #for i in range(10):
    #    test_hand_draw()
    #model =load_my_shallow_net()
    #if (model == None):
    #    print("Failed to load model")
    #else:
        # Call the function with the loaded model and
        #got = predict_data(model, image)
        #print("Predicted digit:", got)
    #    random_test_samples(model, 20)
    test_drawed_digits_shallow()
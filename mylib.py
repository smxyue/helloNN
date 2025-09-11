import pickle
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

import gzip

def load_data(filename="data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return [training_data, validation_data, test_data]

def show_mnist_size():
    training_data, validation_data, test_data = load_data()
    print("Training data size:", len(training_data[0]))
    print("Validation data size:", len(validation_data[0]))
    print("Test data size:", len(test_data[0]))
def get_digital(digit):
    """
    Load the MNIST dataset, find samples of the specified digit from the training set,
    and display them in a 10x8 grid.
    
    Parameters:
    digit (int): The digit to find and display (0-9)
    """
    # Load the data
    training_data, validation_data, test_data = load_data()
    
    # Extract training images and labels
    training_images, training_labels = training_data
    
    # Find indices where the label matches the specified digit
    digit_indices = np.where(training_labels == digit)[0]
    
    # Select up to 80 samples (10x8 grid)
    num_samples = min(80, len(digit_indices))
    selected_indices = digit_indices[:num_samples]
    
    # Create a 10x8 subplot
    fig, axes = plt.subplots(10, 8, figsize=(12, 15))
    fig.suptitle(f'Samples of digit {digit}', fontsize=16)
    

    # Display the images
    start_idx = np.random.randint(0, len(digit_indices) - num_samples)
    
    print(start_idx)
    for i in range(10):
        for j in range(8):
            idx = i * 8 + j
            if idx < num_samples:
                # Get the image data and reshape it to 28x28
                image = training_images[digit_indices[start_idx + idx]].reshape(28, 28)
                axes[i, j].imshow(image, cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
            else:
                # Hide empty subplots
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def match_mean(img, target_mean=0.1307):
    img = img.astype(np.float32)
    current_mean = img.mean()
    img = img - current_mean + target_mean
    # 确保仍在 [0, 1]
    img = np.clip(img, 0, 1)
    return img
def load_image(filename):
    mnist_mean = 0.1307
    mnist_std = 0.3081
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
   
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    
    image = 1-image  # Invert colors if necessary
 
    # Apply MNIST normalization (standardize using MNIST mean and std)
    image = (image - image.mean()) / image.std()
    
    # Adjust to match MNIST's specific mean and std
    image = image * mnist_std + mnist_mean
    
    # Ensure values stay in valid range [0, 1]
    #image = np.clip(image, 0, 1)
    
    return image

def plot_image(image, title="Image"):

    plt.imshow(image, cmap='gray') 
    plt.title(title)
    plt.axis('off') 
    plt.show()


def mnist_image(filename):
    
    
    # Load image in grayscale
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Invert colors (black to white, white to black)
    image = 255 - image
    
    # Find bounding box of non-zero pixels and crop
    coords = np.column_stack(np.where(image > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = image[y_min:y_max+1, x_min:x_max+1]
         # 修正：补齐为正方形
        h, w = cropped.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=cropped.dtype)
        y_start = (size - h) // 2
        x_start = (size - w) // 2
        square[y_start:y_start+h, x_start:x_start+w] = cropped
    else:
        # If image is all zeros, use the original image
        square = image
    
    # Resize to 20x20
    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Create a 28x28 canvas with black background
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # Place the resized image in the center (with 4 pixels border on each side)
    canvas[4:24, 4:24] = resized
    
    # Normalize to [0, 1] range
    normalized = canvas.astype(np.float32) / 255.0
    
    # Apply MNIST normalization
    mnist_mean = 0.1307
    mnist_std = 0.3081
    
    # Standardize using current image statistics
    current_mean = normalized.mean()
    current_std = normalized.std()
    
    if current_std > 0:
        standardized = (normalized - current_mean) / current_std
        # Adjust to match MNIST's specific mean and std
        result = standardized * mnist_std + mnist_mean
    else:
        result = np.full_like(normalized, mnist_mean)
    
    # Ensure values stay in valid range [0, 1]
    result = np.clip(result, 0, 1)
    
    return result



def create_drawing_window():
    """
    Creates a blank white canvas window where user can draw with black pen.
    After finishing, clicking 'Finish' saves the image as 'saved.png'.
    
    Returns:
        str: The filename "saved.png" after saving the drawing.
    """
    
    # Create main window
    root = tk.Tk()
    root.title("Drawing Canvas")
    
    # Canvas dimensions
    canvas_width = 800
    canvas_height = 600
    
    # Create canvas with white background
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()
    
    # PIL image for saving
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    
    # Store last mouse position
    last_x, last_y = None, None
    
    def paint(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y
        if last_x is not None and last_y is not None:
            # Draw on canvas
            canvas.create_line((last_x, last_y, x, y), fill='black', width=8, capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on PIL image
            draw.line([last_x, last_y, x, y], fill='black', width=8)
        last_x, last_y = x, y

    def reset_position(event):
        nonlocal last_x, last_y
        last_x, last_y = None, None

    def save_and_exit():
        # Save image
        image.save("saved.png")
        # Close window
        root.destroy()
    
    # Create finish button
    finish_button = tk.Button(root, text="Finish", command=save_and_exit)
    finish_button.pack()
    
    # Bind mouse events
    canvas.bind('<B1-Motion>', paint)
    canvas.bind('<ButtonRelease-1>', reset_position)
    
    # Start GUI event loop
    root.mainloop()
    
    return "saved.png"

def random_show_mnist_data():
    td,vd,td= load_data()
    dataCount = len(td[0])
    randomIndex = np.random.randint(0, dataCount)
    while randomIndex + 80 > dataCount:
        randomIndex = np.random.randint(0, dataCount)
    

    plt.figure(figsize=(9, 9))
    
    for x in range(8):
        for y in range(10):
            plt.subplot(8,10, x*10 + y + 1)
            dataimg = td[0][randomIndex].reshape(28, 28)
            label = td[1][randomIndex]
            toa = np.array(dataimg)
            img = toa.reshape(28,28)
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.title(f"{label}")
            plt.axis('off')
            randomIndex += 1
    plt.tight_layout()
    plt.show()
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from myConvClass import draw_test_conv
from mylib import create_drawing_window, get_digital, mnist_image, plot_image, random_show_mnist_data, show_mnist_size
from torch_shallow import draw_test, load_data_shared

# Add the path to the directory containing conv.py
sys.path.append("D:/neural-networks-and-deep-learning-master/src")

# Import the module

# Now you can call the function


if __name__ == "__main__":
    get_digital(7)
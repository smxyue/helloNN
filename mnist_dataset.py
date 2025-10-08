import gzip
import pickle
import random
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from mylib import load_data


class MnistDataset(Dataset):
    
    def __init__(self, filename="data/mnist.pkl.gz"):
        f = gzip.open(filename, 'rb')
        self.data_df, self.validation_data, self.test_data = pickle.load(f, encoding="latin1")
        f.close()
    
    def __len__(self):
        return len(self.data_df[0])
    
    def __getitem__(self, index):
        # image target (label)
        label = self.data_df[1][index]
        target = torch.zeros((10))
        target[label] = 1.0
        
        # image data
        image_values = torch.FloatTensor(self.data_df[0][index])
        
        # return label, image data tensor and target tensor
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df[0][index].reshape(28,28)
        plt.title("label = " + str(self.data_df[1][index]) + "index:" +str(index))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()

if __name__ == "__main__":
    dataset = MnistDataset()
    print("len = ", len(dataset))
    dataset.plot_image(random.randint(0, len(dataset)-1))
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# <---------- Custom Dataset Class ---------->
class HeightReconstructionDataset(Dataset):
    def __init__(self, annotations_file, input_tensor_dir, output_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_labels.columns = ['Input Tensor', 'Output Height Data']
        
        self.input_tensor_dir = input_tensor_dir
        self.output_dir = output_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        input_tensor_path = os.path.join(self.input_tensor_dir, self.img_labels.iloc[idx, 0])
        input_tensor = np.load(input_tensor_path)
        height_data_path = os.path.join(self.output_dir, self.img_labels.iloc[idx, 1])
        height_data = np.load(height_data_path)
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            height_data = self.target_transform(height_data)
        return input_tensor, height_data
    
    def get_input_tensor(self, idx):
        input_tensor_path = os.path.join(self.input_tensor_dir, self.img_labels.iloc[idx, 0])
        return np.load(input_tensor_path)
    
    def get_height_data(self, idx):
        height_data_path = os.path.join(self.output_dir, self.img_labels.iloc[idx, 1])
        return np.load(height_data_path)
    
    def show_input(self, idx):
        input_tensor = self.get_input_tensor(idx)
        
        fig, axis = plt.subplots(1,1)
        axis.imshow(input_tensor[:, :, 0]);
        axis.set_title(f"{idx}: Input North");
        axis.set_axis_off();

        fig, axes = plt.subplots(1,2, figsize=(10.2, 10.2))
        axes[0].imshow(input_tensor[:, :, 3]);
        axes[0].set_title(f"{idx}: input West");
        axes[0].set_axis_off();
        axes[1].imshow(input_tensor[:, :, 1])
        axes[1].set_title(f"{idx} Input East");
        axes[1].set_axis_off();

        fig, axis = plt.subplots(1,1)
        axis.imshow(input_tensor[:, :, 2]);
        axis.set_title(f"{idx} Input South");
        axis.set_axis_off();
        
    def show_height(self, idx):
        height_data = self.get_height_data(idx)
        
        plt.imshow(height_data)
        plt.title(f"{idx}: Height Data")
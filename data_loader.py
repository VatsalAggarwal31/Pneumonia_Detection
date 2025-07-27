import numpy as np
import pandas as pd
import os
import torch
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, RAMDOM_SEED, DATA_DIR, METADATA_PATH, OUTPUT_DIR


class ChestXRayDataset:
    def __init__(self, df, img_dir, transform=None):
        self.df = df    #df = DataFrame
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)     #Total Number of samples in the dataset

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['Image Index'])

        image = Image.open(img_name).convert('RGB')

        label = self.df.iloc[idx]['Label']
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)


        return image, label_tensor




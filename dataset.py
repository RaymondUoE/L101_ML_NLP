import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import *

class CustomEmbeddingDataset(Dataset):
    def __init__(self, sentences_file, encoder, device, transform=None, target_transform=None):
        df = pd.read_csv(sentences_file)
        self.labels = torch.tensor([[float(y) for y in string_of_list_to_list(x)] for x in df['label_dist']]).to(device)
        self.ps = encoder.encode(df['premise'], convert_to_tensor=True, show_progress_bar=True).to(device)
        self.hs = encoder.encode(df['hypothesis'], convert_to_tensor=True, show_progress_bar=True).to(device)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        p = self.ps[idx, :]
        h = self.hs[idx, :]
        label = self.labels[idx, :]
        if self.transform:
            p = self.transform(p)
            h = self.transform(h)
        if self.target_transform:
            label = self.target_transform(label)
        return {'p': p,
                'h': h,
                'label': label}
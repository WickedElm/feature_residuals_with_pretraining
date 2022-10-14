import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd

class StandardNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        return flow, label

class SarhanFormatNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, :-1].shape
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, :-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        return flow, label

class SarhanFormatWithCacheNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, :-1].shape
        self.class_weights = class_weights
        self.L_cache = None
        self.S_cache = None

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, :-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        if self.L_cache == None:
            L = torch.zeros_like(flow)
            S = torch.zeros_like(flow)
        else:
            L = self.L_cache[idx]
            S = self.S_cache[idx]

        return flow, L, S, label

class SScalingNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights
        self.L_cache = None
        self.S_cache = None

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        if self.L_cache == None:
            L = torch.zeros_like(flow)
            S = torch.zeros_like(flow)
        else:
            L = self.L_cache[idx]
            S = self.S_cache[idx]

        return flow, L, S, label

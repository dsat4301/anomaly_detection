import numpy as np
import torch
# noinspection PyProtectedMember
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """torch.utils.data.Dataset implementation."""

    def __init__(self, data: np.ndarray):
        self.data = data.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> torch.FloatTensor:
        return torch.FloatTensor(self.data[index, :])

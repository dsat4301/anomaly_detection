import numpy as np
import torch
# noinspection PyProtectedMember
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ torch.utils.data.Dataset implementation, serving as wrapper for an np.ndarray. """

    def __init__(self, data: np.ndarray):
        """
        :param data : np.ndarray
            The array of data.
        """
        self.data = data.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> torch.FloatTensor:
        return torch.FloatTensor(self.data[index, :])

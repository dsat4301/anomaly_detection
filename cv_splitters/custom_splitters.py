import math

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state


class LeaveOutOneClassForTrainingCrossValidator(BaseCrossValidator):

    def __init__(
            self,
            leave_out_class_label: int,
            shuffle: bool = False,
            random_state: int = None):
        self.leave_out_class_label = leave_out_class_label
        self.shuffle = shuffle
        self.random_state = random_state

    # noinspection PyPep8Naming
    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None:
            raise ValueError
        y = np.array(y)

        leave_out_indices = np.argwhere(y == self.leave_out_class_label).flatten()
        fold_size = len(leave_out_indices)
        n_training_samples = len(y) - fold_size

        return math.floor(n_training_samples / fold_size)

    # noinspection PyPep8Naming
    def _iter_test_indices(self, X=None, y=None, groups=None):
        if y is None:
            raise ValueError
        y = np.array(y)

        leave_out_indices = np.argwhere(y == self.leave_out_class_label).flatten()
        fold_size = len(leave_out_indices)
        n_training_samples = len(y) - len(leave_out_indices)
        n_splits = math.floor(n_training_samples / fold_size)

        mask = np.ones(len(y), np.bool)
        mask[leave_out_indices] = 0
        indices = np.argwhere(mask == 1)

        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        fold_sizes = np.full(n_splits, fold_size, dtype=np.int)
        fold_sizes[:n_training_samples % n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield np.append(indices[start:stop], leave_out_indices)
            current = stop

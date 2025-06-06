import os
import numpy as np
import torch

class SequentialBatchLoader:
    def __init__(self, filepath, batch_size, block_size, device, seed=1337):
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.idx = 0
        self.total_tokens = len(self.data)

    def get_batch(self):
        x_batch = []
        y_batch = []

        for _ in range(self.batch_size):
            if self.idx + self.block_size + 1 >= self.total_tokens:
                self.idx = 0  # next epoch

            x = self.data[self.idx : self.idx + self.block_size]
            y = self.data[self.idx + 1 : self.idx + 1 + self.block_size]

            x_batch.append(torch.from_numpy(x.astype(np.int64)))
            y_batch.append(torch.from_numpy(y.astype(np.int64)))

            self.idx += self.block_size  # move window forward (non-overlapping)

        x = torch.stack(x_batch)
        y = torch.stack(y_batch)

        if 'cuda' in str(self.device):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y
import csv

import numpy as np
import random


class MnistDataset:
    def __init__(self, data_file, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(data_file, "r") as file:
            reader = csv.reader(file)
            header = next(reader)
            data = [row for row in reader]

        self.data = np.array(data, dtype=int)
        self.img = self.data[:, 1:] / 256
        self.label = self.data[:, 0]
        self.label = np.squeeze(np.eye(10)[self.label.reshape(-1)])

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        indexes = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indexes)

        n_batches = len(self)
        for i in range(n_batches):
            span = slice(i * self.batch_size, (i + 1) * self.batch_size)
            batch_indexes = indexes[span]
            yield self.img[batch_indexes], self.label[batch_indexes]

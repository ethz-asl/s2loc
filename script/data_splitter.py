import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

class DataSplitter:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        print("[splitter] dataset size: ", dataset_size);
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))
        print("[splitter] test_split: ", test_split);

        if shuffle:
            np.random.shuffle(self.indices)

        print("[splitter] indices size: ", len(self.indices))
        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        print("[splitter] train size: ", train_size)
        print("[splitter] test size: ", len(self.test_indices))

        validation_split = int(np.floor((1 - val_train_split) * train_size))
        print("[splitter] val split: ", validation_split);

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader

    def get_train_size(self):
        return len(self.train_indices)

    def get_val_size(self):
        return len(self.val_indices)

    def get_test_size(self):
        return len(self.test_indices)

if __name__ == "__main__":
    from data_source import DataSource
    from training_set import TrainingSet

    ds = DataSource('/mnt/data/datasets/Spherical/training')
    ds.load(10)
    ts = TrainingSet(ds)

    split = DataSplitter(ts, shuffle=True)

    train_loader, val_loader, test_loader = split.get_split(batch_size=2, num_workers=1)

    for iteration, batch in enumerate(train_loader,1):
        print('{} : {}'.format(iteration, str(len(batch))))

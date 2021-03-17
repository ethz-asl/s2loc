import logging

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler


class DataSplitter:

    def __init__(self, dataset, restore, test_train_split=0.9, val_train_split=0.1, shuffle=False):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if not restore:
            self.split_for_train(test_train_split, val_train_split, shuffle)
        else:
            self.split_for_test()

    def split_for_train(self, test_train_split, val_train_split, shuffle):
        test_split = int(np.floor(test_train_split * self.dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:
                                                        test_split], self.indices[test_split:]
        train_size = len(train_indices)

        validation_split = int(np.floor((1 - val_train_split) * train_size))
        self.train_indices, self.val_indices = train_indices[:
                                                             validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def split_for_test(self):
        self.test_sampler = SequentialSampler(self.dataset)
        self.test_indices = self.dataset.test_indices

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')

        if self.train_sampler != None and self.val_sampler != None:
            self.train_loader = self.get_train_loader(
                batch_size=batch_size, num_workers=num_workers)
            self.val_loader = self.get_validation_loader(
                batch_size=batch_size, num_workers=num_workers)

        self.test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
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

    ds = DataSource('/mnt/data/datasets/Spherical/test_training/', 10)
    ds.load(10)
    ts = TrainingSet(restore=False, bw=100)
    ts.generateAll(ds)

    split = DataSplitter(ts, restore=False, test_train_split=1.0, shuffle=True)

    train_loader, val_loader, test_loader = split.get_split(
        batch_size=2, num_workers=1)
    train_size = split.get_train_size()
    val_size = split.get_val_size()
    test_size = split.get_test_size()
    print("Training size: ", train_size)
    print("Validation size: ", val_size)
    print("Testing size: ", test_size)

    for iteration, batch in enumerate(train_loader, 1):
        print('{} : {}'.format(iteration, str(len(batch))))

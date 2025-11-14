import torch
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        y = self.data[:,-1]
        x = self.data[:,:-1]
        # indices of collision
        idx_collision = np.where(y==1)[0]
        idx_not_collision = np.where(y==0)[0]

        #oversampling minority class
        # max_count = max(len(idx_collision), len(idx_not_collision)) #oversample target size
        # if len(idx_collision) < max_count:
        #     add_0 = np.random.choice(idx_collision, max_count - len(idx_collision), replace=True)
        #     idx_0 = np.concatenate([idx_collision, add_0])
        #
        # if len(idx_not_collision) < max_count:
        #     add_1 = np.random.choice(idx_not_collision, max_count - len(idx_1), replace=True)
        #     idx_1 = np.concatenate([idx_not_collision, add_1])

        #undersampling majority class
        min_count = min(len(idx_collision), len(idx_not_collision)) #undersample target size
        idx_collision_sampled = np.random.choice(idx_collision, min_count, replace=False)
        idx_not_collision_sampled = np.random.choice(idx_not_collision, min_count, replace=False)

        #combine and shuffle
        balanced_idx = np.concatenate([idx_collision_sampled, idx_not_collision_sampled])
        np.random.shuffle(balanced_idx)

        self.data = self.data[balanced_idx]

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        entry = self.normalized_data[idx]
        x = entry[:-1].astype(np.float32)
        y = entry[-1].astype(np.float32)
        return {'input': x, 'label': y}
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.train_size = int(0.8 * len(self.nav_dataset))
        self.test_size = int(len(self.nav_dataset)-self.train_size)

        self.train_dataset, self.test_dataset = random_split(self.nav_dataset, [self.train_size,self. test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()

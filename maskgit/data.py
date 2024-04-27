import torch
import torchvision
# from torchvision.transforms import v2 ####
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os


class CELEBAImageDataset(Dataset):
    def __init__(self, image_paths, image_size):
        self.target_size = image_size
        self.image_paths = image_paths
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)[10:, 2:]
        image = self.to_tensor((image / 127.5 - 1).astype(np.float32))
        return image


class AFHQImageDataset(Dataset):
    def __init__(self, image_paths,image_size):
        self.target_size = image_size
        self.intrp_method = cv2.INTER_LANCZOS4
        self.image_paths = image_paths
        self.to_tensor = torchvision.transforms.ToTensor()
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.target_size, interpolation=self.intrp_method)
        image = self.to_tensor((image / 127.5 - 1).astype(np.float32))
        image = self.flip(image)

        return image


def build_dataset(phase,
                  config,
                  data_path_afhq=None,
                  data_path_celeba=None,
                  data_path_enc=None):
    if phase == 1:
        if 'afhq' in config['ver']:
            split = round(config['dataset_size'] / 3 * config['train_test_split'])
            train_paths = [os.path.join(data_path_afhq, s, f"{i:0>4d}.png") \
                           for i in range(split) \
                           for s in ['cat', 'dog', 'wild']]
            test_paths = [os.path.join(data_path_afhq, s, f"{i:0>4d}.png") \
                          for i in range(split, round(config['dataset_size'] / 3)) \
                          for s in ['cat', 'dog', 'wild']]

            train_data = AFHQImageDataset(train_paths,image_size=config['image_size'])
            test_data = AFHQImageDataset(test_paths,image_size=config['image_size'])

        elif 'celeba' in config['ver']:
            split = round(config['dataset_size'] * config['train_test_split'])
            train_paths = [os.path.join(data_path_celeba, f"{i:0>6d}.jpg") for i in range(1, split + 1)]
            test_paths = [os.path.join(data_path_celeba, f"{i:0>6d}.jpg") for i in
                          range(split + 1, config['dataset_size'] + 1)]

            train_data = CELEBAImageDataset(train_paths,image_size=config['image_size'])
            test_data = CELEBAImageDataset(test_paths,image_size=config['image_size'])

        print(f"train data shape: {len(train_data)}\ntest data shape: {len(test_data)}")

        train_dataloader = DataLoader(train_data, batch_size=config['batch_size'],
                                      shuffle=True, num_workers=4, drop_last=True)

        test_dataloader = DataLoader(test_data, batch_size=config['batch_size']/config['batch_acc'],
                                     shuffle=True, num_workers=4, drop_last=True)

    elif phase == 2:
        ori_data = np.load(data_path_enc).astype('float32')

        config['dataset_size'] = min(config['dataset_size'], ori_data.shape[0])
        split_ind = int(config['dataset_size'] * config['train_test_split'])
        train_data, test_data = ori_data[:split_ind], ori_data[split_ind:config['dataset_size']]
        print(f"train data shape: {train_data.shape}\ntest data shape: {test_data.shape}")

        train_data_tensor = torch.from_numpy(train_data)
        train_dataset = torch.utils.data.TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                      shuffle=True, drop_last=True)

        test_data_tensor = torch.from_numpy(test_data)
        test_dataset = torch.utils.data.TensorDataset(test_data_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size']/config['batch_acc'],
                                     shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader
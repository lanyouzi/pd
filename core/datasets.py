'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-03-06 15:05:10
LastEditTime: 2022-07-07 20:06:52
FilePath: /pd/core/datasets.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import random

transform = transforms.Compose([
    transforms.ToTensor()
])

class SpecDataset(Dataset):
    def __init__(self, data_path, csv_path, num = 50, mode='train') -> None:
        csv_path = os.path.join(csv_path, f'spec{num}', mode)
        subject_list = os.listdir(csv_path)
        # subject_list.sort()
        
        label_list = range(len(subject_list))
        code2int_dict = dict(zip(subject_list, label_list))
        # print(code2int_dict)
        self.images = list()
        self.labels = list()
        self.data_path = data_path
        self.mode = mode

        for subject in subject_list:
            code_path = os.path.join(csv_path, subject)
            df = pd.read_csv(code_path, index_col=0)
            self.images.extend(df['recordPath'])
            self.labels.extend([code2int_dict[subject] for _ in range(len(df))])
        # print(len(self.images), len(self.labels))
        print(f'Initialize spectrogram dataset including {len(self.images)} records of {len(subject_list)} people.')
        assert len(self.images)==len(self.labels)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.images[index])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)
        return tensor, self.labels[index]
    
    def get_labels(self):
        return self.labels

class MedTestingDataset(Dataset):
    def __init__(self, csv_path, data_path, mode='train') -> None:
        self.data_path = data_path
        with open(os.path.join(csv_path, f'{mode}_split.txt')) as f:
            self.images = list()
            self.labels = list()
            for line in f.readlines():
                image1, image2, label = line.strip('\n').split('\t')
                self.images.append((image1, image2))
                self.labels.append(label)
            assert len(self.images)==len(self.labels)
            f.close()
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path1, path2 = self.images[index]
        image1 = Image.open(os.path.join(self.data_path, path1)).convert('RGB')
        image2 = Image.open(os.path.join(self.data_path, path2)).convert('RGB')
        return transform(image1), transform(image2), int(self.labels[index])

if __name__ == '__main__':
    dataset = SpecDataset('/mnt/DataCenter/maruokai/datasets/pd', '/home/maruokai/workplace/pd/subject_detection/user_data')
    # dataset = SpecDataset('/home/maruokai/workplace/pd/data/spec15', mode='valid')
    # dataset = MedTestingDataset('/mnt/DataCenter/maruokai/datasets/pd/feature/feature_seg')

    
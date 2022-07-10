'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-06-06 20:17:56
LastEditors: Please set LastEditors
LastEditTime: 2022-07-10 20:32:23
FilePath: /pd/subject_detection/main.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''
import os
import sys
sys.path.append('./')
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from utils.utils import get_distribution_dataframe, plot_loss_from_log

config_path = 'config/spec.yaml'
with open(config_path, encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
gpu = config['train']['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
csv_dir = config['base']['csv_dir']
data_dir = config['base']['data_dir']
batch_size = config['train']['batch_size']
lr = float(config['train']['lr'])
num_epoch = int(config['train']['epoch'])
subject = int(config['base']['subject'])
use_weight_sampler = config['train']['use_weight_sampler']

import sklearn.metrics as metrics
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from core.train import train, predict, test
from core.datasets import SpecDataset
from core.models import SpecNet, FocalLoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

dataset = SpecDataset(data_path=data_dir, csv_path = csv_dir, num=int(subject), mode='train')
test_dataset = SpecDataset(data_path=data_dir, csv_path = csv_dir, num=int(subject), mode='test')

# resampling
train_size = int(len(dataset)*0.9)
valid_size = len(dataset)-train_size
train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[train_size, valid_size])

target = np.array(dataset.labels)[train_dataset.indices]
class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

if use_weight_sampler:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))

model = SpecNet(out_num=subject)
# criterion = nn.CrossEntropyLoss(reduction='sum')
criterion = FocalLoss(class_num=subject, reduction='sum', device=device)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
checkpoints_dir = train(
    model=model, num_epoch=num_epoch, criterion=criterion, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, optimizer=optimizer, save_dir='subject_detection', device=device)
# checkpoints_dir = '/home/maruokai/workplace/pd/subject_detection/runs/2022-06-23_20:55:39'
plot_loss_from_log(checkpoints_dir)
state = torch.load(os.path.join(checkpoints_dir, 'best_ckpt.pth'))
best_acc = state['acc']
best_epoch = state['epoch']

model = state['model']
y_pred, y_true = predict(model, test_dataloader, device)
print(f'Best acc: {best_acc}, best epoch: {best_epoch}')

save_dir = os.path.join(checkpoints_dir, 'res')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(subject/2, subject/2))
sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu', fmt='g')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(save_dir, 'confusion_matrix'))

# data_distribution
df = get_distribution_dataframe(y_true, y_pred)
plt.figure(figsize=(14, 6))
sns.histplot(data=df, x='label', hue='type', multiple='dodge',
             discrete=True, shrink=.7, kde=True)
plt.title('Data Distribution')
plt.savefig(os.path.join(save_dir, 'data_distribution'))

# classification report
classification_report = metrics.classification_report(y_true, y_pred)
print(classification_report)
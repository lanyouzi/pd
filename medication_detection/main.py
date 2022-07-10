'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-07-05 14:33:15
LastEditTime: 2022-07-08 18:22:25
FilePath: /pd/medication_detection/main.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''
import os
import sys
import random
sys.path.append('./')
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from utils.utils import get_distribution_dataframe, plot_loss_from_log
config_path = 'config/med.yaml'
with open(config_path, encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    
gpu = config['train']['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
csv_dir = config['base']['csv_dir']
data_dir = config['base']['data_dir']
base_model_path = config['base']['base_model_path']
batch_size = config['train']['batch_size']
lr = float(config['train']['lr'])
num_epoch = int(config['train']['epoch'])

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from core.train_med import train, predict, test
from core.datasets import MedTestingDataset
from core.models import SpecNet, MedTestingNet, FocalLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

dataset = MedTestingDataset(csv_path=csv_dir, data_path=data_dir, mode='train')
train_len = int(len(dataset)*0.9)
valid_len = int(len(dataset)-train_len)
train_dataset, valid_dataset = random_split(dataset, lengths=[train_len, valid_len])
test_dataset = MedTestingDataset(csv_path=csv_dir, data_path=data_dir, mode='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))

base_model = torch.load(base_model_path)['model']
model = MedTestingNet(base_model)
# model = MedTestingNet()
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.AdamW(filter(lambda x:x.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
checkpoints_dir = train(model=model, num_epoch=num_epoch, criterion=criterion, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, optimizer=optimizer, save_dir='medication_detection', device=device)
plot_loss_from_log(checkpoints_dir)

base_model = torch.load(base_model_path)['model']
model = MedTestingNet(base_model)
# state = torch.load('/mnt/DataCenter/maruokai/datasets/pd/checkpoint/3/best_ckpt.pth', map_location=torch.device('cuda'))
# model.load_state_dict(state['model'])

logits, label = predict(model, test_dataloader, device=device)
fpr, tpr, thresholds = metrics.roc_curve(y_true=label, y_score=logits)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='orange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='dashed')
plt.plot([0, 1], [1, 0], color='green', lw=lw, linestyle='dotted')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(os.path.join(checkpoints_dir, 'res', 'ROC.png'))

logits_np = np.asarray(logits)

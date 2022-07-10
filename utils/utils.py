'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-03-06 15:05:26
LastEditTime: 2022-06-24 19:10:49
FilePath: /pd/utils/utils.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''
import yaml
import pandas as pd
import numpy as np
import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
import logging

 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    # file logger
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # terminal logger
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


# return a dict that maps record ids to integers which occur most
def get_rid2code(csv_path, subject:int)->dict:
    df = pd.read_csv(csv_path)
    # df['healthCode'].unique()
    unique_code = df['healthCode'].value_counts().iloc[:subject].index
    print(unique_code)
    df = df[df['healthCode'].isin(unique_code)]
    code2int = pd.Series(data=range(len(unique_code)), index=unique_code).to_dict()
    df['healthCode'] = df['healthCode'].map(code2int)
    # print(df['healthCode'], df['recordId'])
    id2code = pd.Series(data=df['healthCode'].values, index=df['recordId'].values).to_dict()
    return id2code


# shuffle features and labels accordingly
def shuffle(X, y):
    assert len(X) == len(y)
    randomize = np.arange(len(y))
    np.random.shuffle(randomize)
    return X[randomize], y[randomize]

def get_distribution_dataframe(y_true, y_pred):
    true_df = pd.DataFrame(zip(y_true, np.ones(len(y_true))), columns=['label', 'num'])
    true_df['type'] = 'label'
    pred_df = pd.DataFrame(zip(y_pred, np.ones(len(y_true))), columns=['label', 'num'])
    pred_df['type'] = 'prediction'
    df = pd.concat([true_df, pred_df])
    df = df.reset_index(drop=True)
    return df

def plot_loss_from_log(log_path):
    train_loss = list()
    train_acc = list()
    valid_loss =list()
    valid_acc = list()
    with open(os.path.join(log_path, 'history.log'), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if 'Epoch' in line[0]:
                # print(line)
                train_loss.append(float(line[1].split('=')[1]))
                train_acc.append(float(line[2].split('=')[1]))
                valid_loss.append(float(line[3].split('=')[1]))
                valid_acc.append(float(line[4].split('=')[1]))
        f.close()
        print(train_loss[:10])
        print(train_acc[:10])
    save_path = os.path.join(log_path, 'res')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(valid_acc, label = 'valid_acc')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(valid_loss, label = 'valid_loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    
    return train_loss, train_acc, valid_loss, valid_acc

if __name__=='__main__':
    plot_loss_from_log('/home/maruokai/workplace/pd/subject_detection/runs/2022-06-24_21:14:44')

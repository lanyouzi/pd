'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-06-17 15:14:55
LastEditTime: 2022-07-07 16:26:14
FilePath: /pd/subject_detection/generate_user_data.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''

import os
import shutil
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import train_test_split
# return a dict that maps record ids to health codes


def rid2code(csv_path) -> dict:
    df = pd.read_csv(csv_path)
    id2code = pd.Series(data=df['healthCode'].values,
                        index=df['recordId'].values).to_dict()
    return id2code

# split train and test folders using spectrogram data in data_dir
def generate_subject_detection_data(save_dir, data_dir, csv_path, subject=10, is_back = False):
    if not is_back:
        prefix_dir = os.path.join('result', 'feature_seg')
    else:
        prefix_dir = os.path.join('result_backward', 'feature_seg')
    data_dir = os.path.join(data_dir, prefix_dir)
    save_dir = os.path.join(save_dir, 'user_data_split')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, f'spec{subject}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'train'))
    os.mkdir(os.path.join(save_dir, 'test'))
    
    health_code_dict = dict() # overall dict that will be saved to csv file later
    data_list = os.listdir(data_dir)
    health_code_list = list()
    rid2code_dict = rid2code(csv_path)
    for record in data_list:
        record_id = record.split('_')[0]
        health_code_list.append(rid2code_dict[record_id])

    # only reverse needed data in data_list and health_code_list
    series = pd.Series(data=health_code_list, index=data_list)
    chosen_code = series.value_counts().iloc[:subject].index
    # print(chosen_code)
    series = series[series.isin(chosen_code)]
    print(len(series), series.value_counts())
    data_list, health_code_list = list(series.index), list(series.values)
     
    # a = [os.path.split(record)[1].split('.')[0].split('_')[0] for record in data_list]
    # s = pd.Series(data=a, index=health_code_list).groupby(level=0)
    # cnt=0
    # print(s.count())
    # for name, group in s:
    #     print(name, len(group.drop_duplicates()))
    #     cnt+= len(group.drop_duplicates())
    # print(len(s), cnt)
    
    # print(data_list[:5], health_code_list[:5])
    for record_file, code in zip(data_list, health_code_list):
        if health_code_dict.get(code) is None:  # first appear
            health_code_dict[code] = list()
        health_code_dict[code].append({
            'healthCode': code,
            'recordId': record_file.split('.')[0].split('_')[0],
            'recordPath': os.path.join(prefix_dir, record_file)
        })
    for code, file in health_code_dict.items():
        df = pd.DataFrame(file)
        test_df = df.sample(n=100, axis=0)
        train_df = df[~df.index.isin(test_df.index)]
        print(len(train_df), len(test_df))
        train_df.to_csv(os.path.join(save_dir, 'train', f'{code}.csv'))
        test_df.to_csv(os.path.join(save_dir, 'test', f'{code}.csv'))


if __name__=='__main__':
    # data_dir = '/mnt/DataCenter/maruokai/datasets/pd'
    # forward_data_dir = os.path.join(data_dir, 'result')
    # backward_data_dir = os.path.join(data_dir, 'result_backward')
    # csv_path = os.path.join(data_dir, 'pd_gait.csv')
    
    generate_subject_detection_data(save_dir='/home/maruokai/workplace/pd/subject_detection', data_dir='/mnt/DataCenter/maruokai/datasets/pd', csv_path='/mnt/DataCenter/maruokai/datasets/pd/pd_gait.csv', subject=50)

'''
Author: mrk-lyz mrk_lanyouzi@yeah.net
Date: 2022-07-07 13:54:56
LastEditTime: 2022-07-07 19:09:13
FilePath: /pd/medication_detection/generate_pairings.py
Description: 

Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
'''
import os
import shutil
import pandas as pd
import numpy as np


def rid2point(csv_path) -> dict:
    df = pd.read_csv(csv_path)
    df['medTimepoint'] = df['medTimepoint'].map(lambda x: 0 if x=='Immediately before Parkinson medication' else 1)
    id2point_dict = pd.Series(data=df['medTimepoint'].values,
                        index=df['recordId'].values).to_dict()
    return id2point_dict
    
def rid2code(csv_path) -> dict:
    df = pd.read_csv(csv_path)
    id2code = pd.Series(data=df['healthCode'].values,
                        index=df['recordId'].values).to_dict()
    return id2code

def generate_medication_detection_data(save_dir, data_dir, csv_path, subject=50 ,is_back = False):
    if not is_back:
        prefix_dir = os.path.join('result', 'feature_seg')
    else:
        prefix_dir = os.path.join('result_backward', 'feature_seg')
    data_dir = os.path.join(data_dir, prefix_dir)
    save_dir = os.path.join(save_dir, 'user_data_split')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    health_code_dict = dict() # overall dict that will be saved to csv file later
    data_list = os.listdir(data_dir)
    health_code_list = list()
    med_point_list = list()
    rid2code_dict = rid2code(csv_path)
    rid2point_dict = rid2point(csv_path)
    for record in data_list:
        record_id = record.split('_')[0]
        health_code_list.append(rid2code_dict[record_id])
        med_point_list.append(rid2point_dict[record_id])

    # only reverse needed data in data_list and health_code_list
    df = pd.DataFrame({'fileName':data_list, 'healthCode':health_code_list, 'medTimepoint':med_point_list})
    
    train_chosen_code = df['healthCode'].value_counts().iloc[:subject].sample(int(subject*0.9)).index
    test_chosen_code = df['healthCode'].value_counts().iloc[:subject][~df['healthCode'].value_counts().iloc[:subject].index.isin(train_chosen_code)].index
    train_df = df[df['healthCode'].isin(train_chosen_code)]
    test_df = df[df['healthCode'].isin(test_chosen_code)]
    print(len(train_chosen_code), len(test_chosen_code))
    
    print(len(train_df), len(test_df))
    
    with open(os.path.join(save_dir, 'train_split.txt'), 'w') as f:
        for index, sub_df in train_df.groupby(['healthCode']):
            before_df = sub_df[sub_df['medTimepoint']==0]
            after_df = sub_df[sub_df['medTimepoint']==1]
            print(index, len(before_df), len(after_df))
            if len(before_df)==0 or len(after_df)==0:
                break
            # print(before_df.head(), after_df.head())
            max_length = max(len(after_df), len(before_df))
            # write positive samples
            before_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            after_idx = np.random.choice(after_df.index, size=max_length, replace=True)
            for a, b in zip(before_idx, after_idx):
                path1 = os.path.join(prefix_dir, before_df.loc[a]['fileName'])
                path2 = os.path.join(prefix_dir, after_df.loc[b]['fileName'])
                f.write(f'{path1}\t{path2}\t1\n')
            # write negative samples
            before1_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            before2_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            
            for a, b in zip(before1_idx, before2_idx):
                path1 = os.path.join(prefix_dir, before_df.loc[a]['fileName'])
                path2 = os.path.join(prefix_dir, before_df.loc[b]['fileName'])
                f.write(f'{path1}\t{path2}\t0\n')

        f.close()
        
    with open(os.path.join(save_dir, 'test_split.txt'), 'w') as f:
        for index, sub_df in test_df.groupby(['healthCode']):
            before_df = sub_df[sub_df['medTimepoint']==0]
            after_df = sub_df[sub_df['medTimepoint']==1]
            if len(before_df)==0 or len(after_df)==0:
                break
            max_length = max(len(after_df), len(before_df))
            # write positive samples
            before_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            after_idx = np.random.choice(after_df.index, size=max_length, replace=True)
            for a, b in zip(before_idx, after_idx):
                path1 = os.path.join(prefix_dir, before_df.loc[a]['fileName'])
                path2 = os.path.join(prefix_dir, after_df.loc[b]['fileName'])
                f.write(f'{path1}\t{path2}\t1\n')
            
            before1_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            before2_idx = np.random.choice(before_df.index, size=max_length, replace=True)
            
            for a, b in zip(before1_idx, before2_idx):
                path1 = os.path.join(prefix_dir, before_df.loc[a]['fileName'])
                path2 = os.path.join(prefix_dir, before_df.loc[b]['fileName'])
                f.write(f'{path1}\t{path2}\t0\n')

        f.close()
    
    # data_list, health_code_list = list(series.index), list(series.values)
if __name__=='__main__':
    # data_dir = '/mnt/DataCenter/maruokai/datasets/pd'
    # forward_data_dir = os.path.join(data_dir, 'result')
    # backward_data_dir = os.path.join(data_dir, 'result_backward')
    # csv_path = os.path.join(data_dir, 'pd_gait.csv')
    
    generate_medication_detection_data(save_dir='/home/maruokai/workplace/pd/medication_detection', data_dir='/mnt/DataCenter/maruokai/datasets/pd', csv_path='/mnt/DataCenter/maruokai/datasets/pd/pd_gait.csv', subject=50)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
from utils.utils import get_logger

def do_epoch(model, dataloader, criterion, device, optim=None):
    total_loss = 0
    total_correct = 0
    total_num = 0
    for x1, x2, y_true in tqdm(dataloader, leave=False):
        x1, x2, y_true = x1.to(device), x2.to(device), y_true.to(torch.float32).to(device)
        y_pred = model(x1, x2)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_correct += (torch.round(y_pred) == y_true).sum().item()
        total_num+=y_pred.size(0)
    mean_loss = 1.*total_loss / total_num
    mean_accuracy = 1.*total_correct / total_num

    return mean_loss, mean_accuracy

def train(model, num_epoch, criterion, optimizer, train_dataloader, valid_dataloader=None, device='cpu', save_dir='./', save_interval=10, local_path=None):
    begin_epoch = 0
    save_dir = os.path.join(save_dir, 'runs')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if local_path is not None:       # train from the latest checkpoint
        save_dir = os.path.join(save_dir, str(local_path))
        if not os.path.exists(save_dir):
            raise ValueError(f'The cache directory of {save_dir} is NOT exists')
        for cache in os.listdir(save_dir):
            if cache.startswith('epoch') and begin_epoch<int(cache.split('_')[1]):
                begin_epoch = int(cache.split('_')[1])
                state = torch.load(os.path.join(save_dir, cache))
        model.load_state_dict(state['model'])
        tqdm.write(f'Load checkpoint from {save_dir}, and training will begin in epoch {begin_epoch}')
    else:                       # begin a new training phase
        save_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        save_dir = os.path.join(save_dir, str(save_time))
        os.mkdir(save_dir)
        tqdm.write(f'Training begins and models will be saved in path {save_dir}')
    
    model = model.to(device)
    best_acc = 0
    best_epoch = 0
    # train_loss_history, valid_loss_history, train_acc_history, valid_acc_history = [], [], [], []
    
    logger = get_logger(os.path.join(save_dir, 'history.log'))
    logger.info('start training!')
    for epoch in range(1+begin_epoch, num_epoch+1+begin_epoch):
        model.train()
        train_loss, train_acc = do_epoch(model, train_dataloader, criterion, device, optim=optimizer)
        # train_loss_history.append(train_loss)
        # train_acc_history.append(train_acc)
        
        if valid_dataloader is not None:
            model.eval()
            valid_loss, valid_acc = do_epoch(model, valid_dataloader, criterion, device, optim=None)
            # valid_loss_history.append(valid_loss)
            # valid_acc_history.append(valid_acc)
        
        log_info = f'Epoch:[{epoch:03d}/{num_epoch+begin_epoch:03d}]\t train_loss={train_loss:.5f}\t train_acc={train_acc:.4f} \t valid_loss={valid_loss:.5f}\t valid_acc={valid_acc:.4f}'
        logger.info(log_info)
        # save temporary model
        if (epoch%save_interval==0):
            state = {
                'model': model.state_dict(),    # only parameters
                'epoch': epoch
            }
            torch.save(state, os.path.join(save_dir, f'epoch_{epoch}_loss_{valid_loss:.4f}_acc_{valid_acc:.4f}.pth'))
        
        # record best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            state = {
                'model': model,         # save both architecture and parameters
                'acc': valid_acc,
                'loss': valid_loss,
                'epoch': epoch
            }
            torch.save(state, os.path.join(save_dir, 'best_ckpt.pth'))

    logger.info(f'best accuracy: {best_acc:.4f}, best epoch: {best_epoch:03d}')
    logger.info('finish training!')    
    # return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history
    return save_dir


def predict(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    y_pred = list()
    label = list()
    for x, y_true in tqdm(dataloader, leave=False):
        label.extend(y_true.cpu().numpy())
        x, y_true = x.to(device), y_true.to(device)
        with torch.no_grad():
            logits = model(x)
        y_pred.extend(logits.argmax(dim=-1).cpu().numpy())
    return y_pred, label 

def predict(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    y_pred = list()
    label = list()
    for x1, x2, y_true in tqdm(dataloader, leave=False):
        label.extend(y_true.cpu().numpy())
        x1,x2, y_true = x1.to(device), x2.to(device),y_true.to(torch.float32).to(device)
        with torch.no_grad():
            logits = model(x1, x2)
        y_pred.extend(logits.cpu().numpy())
    return y_pred, label 

def test(model, dataloader, criterion, device='cpu'):
    model = model.to(device)
    model.eval()
    test_loss = 0
    test_correct = 0
    total_num = 0
    for x1, x2, y_true in tqdm(dataloader, leave=False):
        x1,x2, y_true = x1.to(device), x2.to(device),y_true.to(device)
        with torch.no_grad():
            logits = model(x1, x2)
        loss = criterion(logits, y_true)
        test_loss+=loss.item()
        test_correct+=(logits.argmax(dim=-1) == y_true).sum().item()
        total_num +=logits.size(0)
    loss = 1.*test_loss/total_num
    acc = 1.*test_correct/total_num
    print(f"Test loss = {loss:.5f}, acc = {acc:.5f}")
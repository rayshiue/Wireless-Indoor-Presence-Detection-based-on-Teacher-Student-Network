# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:36:35 2021

@author: Mint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR

import os
import math
import copy
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# load own function
import data_utils
import TS_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
CUDA_LAUNCH_BLOCKING=1

#%% def funtion
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def validation(model, t_STI, t_amp, t_target, valid_dataloader, epoch_u, n):
    CELoss = nn.CrossEntropyLoss()
    model.eval()
    total_acc = 0
    total_loss = 0
    t_acc = 0
    with torch.no_grad():
        # if epoch_u + 1 == EPOCHS & n == 2:
        #     for j in range(4):
        #         t_u_sti_re = t_STI[t_target == j].transpose(1,2)
        #         t_u_amp_re = t_amp[t_target == j].transpose(1,2)
        #         fig2 = plt.figure(j)
        #         ax3 = fig2.add_subplot(2,1,1, title = 'STI')
        #         ax4 = fig2.add_subplot(2,1,2, title = 'amp')
        #         for i in range(50):
        #             ax3.plot(np.arange(0,224),t_u_sti_re[0,:,i].cpu().numpy())
        #             ax4.plot(np.arange(0,224),t_u_amp_re[0,:,i].cpu().numpy())
        
        t_output = model(t_STI, t_amp)
        del t_STI, t_amp
        t_acc = (t_output.argmax(1) == t_target).sum().item()
        t_acc = t_acc/len(t_target)
        del t_target, t_output
        
        # for idx, (sti, AMP, target) in enumerate(train_dataloader):
        #     output = model(sti, AMP)
        #     del sti, AMP
        #     acc = (output.argmax(1) == target).sum().item()
        #     del output, target
        #     total_acc += acc
        #     del acc
        # t_acc = total_acc/len(train_dataloader.dataset)
        # total_acc = 0
        for idx, (sti, AMP, target) in enumerate(valid_dataloader):
            output = model(sti, AMP)
            del sti, AMP
            loss = CELoss(output, target)
            total_loss += loss.item()
            del loss
            acc = (output.argmax(1) == target).sum().item()
            del output, target
            total_acc += acc
            del acc
        avg_acc = total_acc/len(valid_dataloader.dataset)
        avg_loss = total_loss/len(valid_dataloader.dataset)
    return avg_acc, avg_loss, t_acc

def cf_matrix(model, sti, AMP, target):
    model.eval()
    # outarray = np.zeros(0)
    # tararray = np.zeros(0)
    with torch.no_grad():
        output = model(sti, AMP)
        del sti
        output = output.argmax(1).data.cpu().numpy()
        target = target.data.cpu().numpy()
        # for idx, (sti, AMP, target) in enumerate(dataloader):
        #     output = model(sti, AMP)
        #     del sti, AMP
        #     output = output.argmax(1).data.cpu().numpy()
        #     target = target.data.cpu().numpy()
        #     np.append(outarray, output)
        #     np.append(tararray, target)
        
        # index = (output != target)
        # incorrect = AMP[index]
        # incorrect = incorrect.transpose(1,2)
        # incorrect_target = target[index]
        # incorrect_output = output[index]
        # for j in range(len(incorrect)):
        #     fig = plt.figure(j)
        #     ax = fig.add_subplot(1,1,1, title = 'amp')
        #     ax.plot(np.arange(0,224),incorrect[j,:,:].cpu().numpy())
        #     plt.xlabel('%d %d'%(incorrect_target[j], incorrect_output[j]))
    matrix = confusion_matrix(target, output)
    return matrix

def save_model(MODEL_PATH):
    MODEL_S_NAME = 'student.pt'
    MODEL_T_NAME = 'teacher.pt'
    
    torch.save({'model_state_dict':model_S.state_dict(),
                }, MODEL_PATH + 'final_' + MODEL_S_NAME)
    print("final student model saved")
    
    torch.save({'model_state_dict':model_T.state_dict(),
                }, MODEL_PATH + 'final_' + MODEL_T_NAME)
    print("final teacher model saved")

    torch.save({'model_state_dict':best_label_model_S.state_dict(),
                }, MODEL_PATH + 'best_acc_label_' + MODEL_S_NAME)
    print("best acc label student model saved")
    
    torch.save({'model_state_dict':best_unlabel_model_S.state_dict(),
                }, MODEL_PATH + 'best_acc_unlabel_' + MODEL_S_NAME)
    print("best acc unlabel student model saved")
    
    torch.save({'model_state_dict':best_unlabel_model_T.state_dict(),
                }, MODEL_PATH + 'best_acc_unlabel_' + MODEL_T_NAME)
    print("best acc unlabel teacher model saved")

def save_plot(MODEL_PATH, EPOCHS, t_list, v_list):
    t_acc_list, t_loss_list = t_list
    v_acc_list, v_loss_list = v_list
    epoch = [i for i in range(EPOCHS)]
    fig1 = plt.figure(1)
    ax1 = plt.subplot(211, title = 'accuracy on label data')
    ax1.plot(epoch, t_acc_list[0], label = 'T train acc' + ' = %.2f' %t_acc_list[0][-1])
    ax1.plot(epoch, t_acc_list[1], label = 'S train acc' + ' = %.2f' %t_acc_list[1][-1])
    ax1.plot(epoch, v_acc_list[0], label = 'T valid acc' + ' = %.2f' %v_acc_list[0][-1])
    ax1.plot(epoch, v_acc_list[1], label = 'S valid acc' + ' = %.2f' %v_acc_list[1][-1])
    lg1 = ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.ylabel('Accuracy')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1, title = 'accuracy on unlabel data')
    ax2.plot(epoch, t_acc_list[2], label = 'T train acc' + ' = %.2f' %t_acc_list[2][-1])
    ax2.plot(epoch, t_acc_list[3], label = 'S train acc' + ' = %.2f' %t_acc_list[3][-1])
    ax2.plot(epoch, v_acc_list[2], label = 'T valid acc' + ' = %.2f' %v_acc_list[2][-1])
    ax2.plot(epoch, v_acc_list[3], label = 'S valid acc' + ' = %.2f' %v_acc_list[3][-1])
    lg2 = ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    plt.show()
    fig1.savefig(MODEL_PATH + 'Accuracy.png', bbox_extra_artists=(lg1,lg2), bbox_inches='tight')
    
    fig2 = plt.figure(2)
    ax1 = plt.subplot(211, title = 'train loss')
    ax1.plot(epoch, t_loss_list[0], label = 'T total loss')
    ax1.plot(epoch, t_loss_list[1], label = 'S Feedback')
    ax1.plot(epoch, t_loss_list[2], label = 'T label loss')
    ax1.plot(epoch, t_loss_list[3], label = 'S unlabel loss')
    ax1.plot(epoch, t_loss_list[4], label = 'T unlabel loss')
    ax1.plot(epoch, t_loss_list[5], label = 'S label loss')
    lg1 = ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.ylabel('Loss')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(212, sharex=ax1, title= 'validation loss')
    ax2.plot(epoch, v_loss_list[0], label = 'T label loss')
    ax2.plot(epoch, v_loss_list[1], label = 'S label loss')
    ax2.plot(epoch, v_loss_list[2], label = 'T unlabel loss')
    ax2.plot(epoch, v_loss_list[3], label = 'S unlabel loss')
    lg2 = ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.show()
    fig2.savefig(MODEL_PATH + 'Loss.png', bbox_extra_artists=(lg1,lg2), bbox_inches='tight')

def save_cm(MODEL_PATH, cf_matrix, title):
    class_names = ['empty','A room','B room','both room']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)
    plt.figure(figsize = (9,6))
    plt.title(title)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig(MODEL_PATH + title + ".png")

def split_valid(sti, AMP, target, valid_ratio = 0.2):
    sti = torch.tensor(sti).type(torch.FloatTensor).to(device)
    AMP = torch.tensor(AMP).type(torch.FloatTensor).to(device)
    target = torch.tensor(target).type(torch.LongTensor).to(device)
    
    # t_u_sti_re = sti.view(-1,50,4,56)
    # t_u_sti_re = sti.transpose(1,2)
    # t_u_amp_re = AMP.view(-1,50,4,56)
    # t_u_amp_re = AMP.transpose(1,2)
    # plt.ion()
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(2,1,1, title = 'STI')
    # ax2 = fig.add_subplot(2,1,2, title = 'amp')
    # for i in range(224):
    #     # ax1.xlabel('subcarrier')
    #     # ax1.ylabel('STI')
    #     ax1.plot(np.arange(0,50),sti[0,:,i].cpu().numpy())
    #     # ax2.xlabel('subcarrier')
    #     # ax2.ylabel('amplitude')
    #     ax2.plot(np.arange(0,50),AMP[0,:,i].cpu().numpy())
    #     plt.pause(0.05)
    # plt.ioff()
    
    # plt.ion()    
    # t_u_sti_re = sti.transpose(1,2)
    # t_u_amp_re = AMP.transpose(1,2)
    # fig2 = plt.figure(2)
    # ax3 = fig2.add_subplot(2,1,1, title = 'STI')
    # ax4 = fig2.add_subplot(2,1,2, title = 'amp')
    # for i in range(50):
    #     # ax3.xlabel('subcarrier')
    #     # ax3.ylabel('STI')
    #     ax3.plot(np.arange(0,224),t_u_sti_re[0,:,i].cpu().numpy())
    #     # ax4.xlabel('subcarrier')
    #     # ax4.ylabel('amplitude')
    #     ax4.plot(np.arange(0,224),t_u_amp_re[0,:,i].cpu().numpy())
    #     plt.pause(1)
    # plt.ioff()
    
    dataset = Data.TensorDataset(sti, AMP, target)
    num_data = len(dataset)
    num_train = round(num_data * (1-valid_ratio))
    num_val = num_data - num_train
    t_dataset, v_dataset = Data.random_split(dataset, [num_train, num_val])
    return t_dataset, v_dataset

#%% train process
def train(Hyper, Teacher, Student, Dataset):
    CELoss = nn.CrossEntropyLoss()
    t_l_iter = iter(Dataset['t']['l']['loader'])
    t_u_iter = iter(Dataset['t']['u']['loader'])
    t_acc_list, t_loss_list = np.zeros((4,Hyper['EPOCHS'])), np.zeros((6,Hyper['EPOCHS']))
    v_acc_list, v_loss_list = np.zeros((4,Hyper['EPOCHS'])), np.zeros((4,Hyper['EPOCHS']))
    bar = tqdm(range(Hyper['total_steps_S']))
    len_l, len_u = len(Dataset['t']['l']['loader'].dataset), len(Dataset['t']['u']['loader'].dataset)
    epoch_l, epoch_u = 0, 0
    total_h = 0
    total_loss_T, total_loss_l_T, total_loss_u_T = 0, 0, 0
    total_loss_S_l_new = 0
    total_loss_S = 0
    
    for step in bar:
        Teacher['model'].train()
        Student['model'].train()

        Teacher['opti'].zero_grad()
        Student['opti'].zero_grad()

        try:
            t_l_sti, t_l_amp, t_l_target = t_l_iter.next()
        except:
            epoch_l += 1
            t_l_iter = iter(Dataset['t']['l']['loader'])
            t_l_sti, t_l_amp, t_l_target = t_l_iter.next()
        try:
            t_u_sti, t_u_amp, t_u_target = t_u_iter.next()
        except:
            epoch_u += 1
            t_u_iter = iter(Dataset['t']['u']['loader'])
            t_u_sti, t_u_amp, t_u_target = t_u_iter.next()
            
        #calculate loss for teacher
        with amp.autocast(enabled=True):
            output_l_T = Teacher['model'](t_l_sti, t_l_amp)
            output_u_T = Teacher['model'](t_u_sti, t_u_amp)
            loss_l_T = CELoss(output_l_T, t_l_target)
            
            soft_pseudo_label = torch.softmax(output_u_T.detach(), dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            
            # max_prob, hard_pseudo_label = torch.max(output_u_T, dim=-1)
            loss_u_T = F.cross_entropy(output_u_T, hard_pseudo_label)
            
            #training student with unlabel data (meta learning)
            output_u_S = Student['model'](t_u_sti, t_u_amp)
            
            loss_S = CELoss(output_u_S, hard_pseudo_label)
            loss_S_u_old = F.cross_entropy(output_u_S.detach(), hard_pseudo_label)
            
        
        Student['scaler'].scale(loss_S).backward()
        Student['scaler'].step(Student['opti'])
        scale_S = Student['scaler'].get_scale()
        Student['scaler'].update()

        with amp.autocast(enabled=True):
            with torch.no_grad():
                output_l_S = Student['model'](t_l_sti, t_l_amp)

            loss_S_l_new = F.cross_entropy(output_l_S.detach(), t_l_target)
            # h = Hyper['LR_S']*(loss_S_u_old - loss_S_l_new)
            # lr_S = Student['sche'].get_last_lr()
            h = (loss_S_l_new - loss_S_u_old)
            loss_T = h*loss_u_T + loss_l_T
        Teacher['scaler'].scale(loss_T).backward()
        Teacher['scaler'].step(Teacher['opti'])
        scale_T = Teacher['scaler'].get_scale()
        Teacher['scaler'].update()
        
        # if not (scale_T != Teacher['scaler'].get_scale()):
        #     Teacher['sche'].step()
        # if not (scale_S != Student['scaler'].get_scale()):
        #     Student['sche'].step()
#==============================================================================
        total_loss_T += loss_T.item()
        total_h += h.item()
        total_loss_u_T += loss_u_T.item()
        total_loss_l_T += loss_l_T.item()
        total_loss_S_l_new += loss_S_l_new.item()
        total_loss_S += loss_S.item()
        
        
        if (step+1) % math.ceil(len_u/Hyper['BS_u']) == 0:
            if not (scale_T != Teacher['scaler'].get_scale()):
                Teacher['sche'].step()
            if not (scale_S != Student['scaler'].get_scale()):
                Student['sche'].step()
            t_loss_list[0,epoch_u] = total_loss_T/(len_l*(Hyper['total_steps_S']/Hyper['total_steps_T']))
            t_loss_list[1,epoch_u] = total_h/len_u
            t_loss_list[2,epoch_u] = total_loss_l_T/(len_l*(Hyper['total_steps_S']/Hyper['total_steps_T']))
            t_loss_list[3,epoch_u] = total_loss_S/len_u
            t_loss_list[4,epoch_u] = total_loss_u_T/len_u
            t_loss_list[5,epoch_u] = total_loss_S_l_new/(len_l*(Hyper['total_steps_S']/Hyper['total_steps_T']))
            #validation
            v_l_acc_T, v_l_loss_T, t_l_acc_T = validation(Teacher['model'], 
                                                          Dataset['t']['l']['sti'], 
                                                          Dataset['t']['l']['amp'],
                                                          Dataset['t']['l']['target'],
                                                          Dataset['v']['l']['loader'],
                                                          epoch_u, 1)
            v_u_acc_T, v_u_loss_T, t_u_acc_T = validation(Teacher['model'],
                                                          Dataset['t']['u']['sti'], 
                                                          Dataset['t']['u']['amp'],
                                                          Dataset['t']['u']['target'],
                                                          Dataset['v']['u']['loader'],
                                                          epoch_u, 2)

            v_l_acc_S, v_l_loss_S, t_l_acc_S = validation(Student['model'],
                                                          Dataset['t']['l']['sti'], 
                                                          Dataset['t']['l']['amp'],
                                                          Dataset['t']['l']['target'],
                                                          Dataset['v']['l']['loader'],
                                                          epoch_u, 3)
            v_u_acc_S, v_u_loss_S, t_u_acc_S = validation(Student['model'],
                                                          Dataset['t']['u']['sti'], 
                                                          Dataset['t']['u']['amp'],
                                                          Dataset['t']['u']['target'],
                                                          Dataset['v']['u']['loader'],
                                                          epoch_u, 4)

            t_acc_list[0,epoch_u] = t_l_acc_T
            t_acc_list[1,epoch_u] = t_l_acc_S
            t_acc_list[2,epoch_u] = t_u_acc_T
            t_acc_list[3,epoch_u] = t_u_acc_S
            
            v_acc_list[0,epoch_u] = v_l_acc_T
            v_acc_list[1,epoch_u] = v_l_acc_S
            v_acc_list[2,epoch_u] = v_u_acc_T
            v_acc_list[3,epoch_u] = v_u_acc_S
            
            v_loss_list[0,epoch_u] = v_l_loss_T
            v_loss_list[1,epoch_u] = v_l_loss_S
            v_loss_list[2,epoch_u] = v_u_loss_T
            v_loss_list[3,epoch_u] = v_u_loss_S
                
            # if v_l_acc_S > best_v_l_acc_S:
            #     best_v_l_acc_S = v_l_acc_S
            #     # best_v_l_loss_S = v_l_loss_S
            #     best_model = copy.deepcopy(model_S)
            #     best_acc_l_model_S = (best_model, v_l_acc_S, v_l_loss_S)
            
            # if v_u_acc_S > best_v_u_acc_S:
            #     best_v_u_acc_S = v_u_acc_S
            #     # best_v_loss_S_u = v_loss_S_u
            #     best_model = copy.deepcopy(model_S)
            #     best_acc_u_model_S = (best_model, v_u_acc_S, v_loss_S_u)
                
            # if v_u_acc_T > best_v_u_acc_T:
            #     best_v_u_acc_T = v_u_acc_T
            #     # best_v_loss_T_u = v_loss_T_u
            #     best_model = copy.deepcopy(model_T)
            #     best_acc_u_model_T = (best_model, v_u_acc_T, v_loss_T_u)
            
            bar.set_description(
                f"S t_l acc: {t_l_acc_S:.3f} S t_u acc: {t_u_acc_S:.3f} "
                f"S v_l acc: {v_l_acc_S:.3f} S v_u acc: {v_u_acc_S:.3f} "
                f"T t_u acc: {t_u_acc_T:.3f} T v_u acc: {v_u_acc_T:.3f} ")      

            total_h = 0
            total_loss_T, total_loss_l_T, total_loss_u_T = 0, 0, 0
            total_loss_S_l_new = 0
            total_loss_S = 0

    
    return (t_acc_list, t_loss_list), (v_acc_list, v_loss_list), Teacher['model'], Student['model']
    # return (t_acc_list, t_loss_list), (v_acc_list, v_loss_list), (best_acc_label_model_S, best_acc_unlabel_model_S, best_acc_unlabel_model_T)

#%% load data
if __name__ == '__main__':
    #L: labeled data     U: unlabeled data 
    BATCH_SIZE_L = 256
    BATCH_SIZE_U = 256
    
    EPOCHS = 50
    
    # Learning rate for teacher & student respectively
    LR_T = 0.015
    LR_S = 0.015
    
    MODEL_PATH = f'model({LR_T}_{LR_S})/'
    try:
        os.mkdir(MODEL_PATH)
        print("Directory " , MODEL_PATH ,  " Created ")
        setup_seed(0)
        # Load labeled data
        DATA_PATH = '20210531_nycu_toilet/csi.db'
        scene_num = 1
        timesteps = 50
        valid_ratio = 0
        
        # load_data will return   train_data, valid_data, train_target, valid_target
        # the shape of data is (7800, 50, 448)
        t_l_data, _, t_l_target, _ = data_utils.load_data(DATA_PATH, scene_num, timesteps, valid_ratio)
        
        # t_l_data = np.transpose(t_l_data,(0,2,1))
        # print(t_l_data.shape)
        
        # Load unlabeled data
        DATA_PATH = '20210604_nycu_toilet/csi.db'
        scene_num = 1
        timesteps = 50
        valid_ratio = 0
        t_u_data, _, t_u_target, _ = data_utils.load_data(DATA_PATH, scene_num, timesteps, valid_ratio)
        # t_u_data = np.transpose(t_u_data,(0,2,1))
        # print(t_u_data.shape)
        
        
        t_l_sti = t_l_data[:,:,:224]  #For all elements in dim1 and dim2, loading the elements from index 0 to index 223 in dim3
        t_l_amp = t_l_data[:,:,224:]  #For all elements in dim1 and dim2, loading the elements from index 224 to the last element in dim3
        del t_l_data
        
        t_u_sti = t_u_data[:,:,:224]
        t_u_amp = t_u_data[:,:,224:]
        del t_u_data
        
#%% pack data  
        #split valid
        t_l_dataset, v_l_dataset = split_valid(t_l_sti, t_l_amp, t_l_target, 0.2)
        t_u_dataset, v_u_dataset = split_valid(t_u_sti, t_u_amp, t_u_target, 0.2)
        
        t_l_sti, t_l_amp, t_l_target = t_l_dataset[:]
        t_u_sti, t_u_amp, t_u_target = t_u_dataset[:]    
        v_l_sti, v_l_amp, v_l_target = v_l_dataset[:]
        v_u_sti, v_u_amp, v_u_target = v_u_dataset[:]
        
        
        total_steps_T = math.ceil(len(t_l_target)/BATCH_SIZE_L) * EPOCHS
        total_steps_S = math.ceil(len(t_u_target)/BATCH_SIZE_U) * EPOCHS
        
        Hyper = {'BS_l':BATCH_SIZE_L,
                  'BS_u':BATCH_SIZE_U,
                  'EPOCHS':EPOCHS,
                  'LR_T':LR_T,
                  'LR_S':LR_S,
                  'total_steps_T':total_steps_T,
                  'total_steps_S':total_steps_S}
    
        t_l_dataloader = Data.DataLoader(t_l_dataset, batch_size=BATCH_SIZE_L, shuffle=True)
        t_u_dataloader = Data.DataLoader(t_u_dataset, batch_size=BATCH_SIZE_U, shuffle=True)
        v_l_dataloader = Data.DataLoader(v_l_dataset, batch_size=BATCH_SIZE_L, shuffle=False)
        v_u_dataloader = Data.DataLoader(v_u_dataset, batch_size=BATCH_SIZE_U, shuffle=False)
        
        Dataset = {'t':{'l':{'loader':t_l_dataloader,'sti':t_l_sti,'amp':t_l_amp,'target':t_l_target},
                        'u':{'loader':t_u_dataloader,'sti':t_u_sti,'amp':t_u_amp,'target':t_u_target}},
                    'v':{'l':{'loader':v_l_dataloader,'sti':v_l_sti,'amp':v_l_amp,'target':v_l_target},
                        'u':{'loader':v_u_dataloader,'sti':v_u_sti,'amp':v_u_amp,'target':v_u_target}}}
        
        # Dataset = {'t':{'l':{'loader':t_l_dataloader},'u':{'loader':t_u_dataloader}},
        #             'v':{'l':{'loader':v_l_dataloader},'u':{'loader':v_u_dataloader}}}
        
#%% model
        # model    
        model_T = TS_model.Teacher().to(device)
        model_S = TS_model.Student().to(device)
        
        scaler_T = amp.GradScaler()
        scaler_S = amp.GradScaler()
        
        # opti_T = torch.optim.Adadelta(model_T.parameters(), lr = LR_T, rho=0.9, weight_decay=0.00001)
        # opti_S = torch.optim.Adadelta(model_S.parameters(), lr = LR_S, rho=0.9, weight_decay=0.00001)
        opti_T = torch.optim.Adam(model_T.parameters(), lr = LR_T)
        opti_S = torch.optim.Adam(model_S.parameters(), lr = LR_S)
        # opti_T = torch.optim.SGD(model_T.parameters(), lr = LR_T, weight_decay=1e-4, momentum=0.8)
        # opti_S = torch.optim.SGD(model_S.parameters(), lr = LR_S, weight_decay=1e-4, momentum=0.8)
        
        sche_T = None
        sche_S = None
        sche_T = torch.optim.lr_scheduler.StepLR(opti_T, step_size=30, gamma=0.5)
        sche_S = torch.optim.lr_scheduler.StepLR(opti_S, step_size=30, gamma=0.5)
        # sche_T = torch.optim.lr_scheduler.OneCycleLR(opti_T, max_lr = LR_T, total_steps = total_steps_S)
        # sche_S = torch.optim.lr_scheduler.OneCycleLR(opti_S, max_lr = LR_S, total_steps = total_steps_S)
        
        Teacher = {'model':model_T,'scaler':scaler_T,'opti':opti_T,'sche':sche_T}
        
        Student = {'model':model_S,'scaler':scaler_S,'opti':opti_S,'sche':sche_S}
        
#%% train
        t_list, v_list, model_T, model_S = train(Hyper, Teacher, Student, Dataset)
        # confusion matrix
        cf_matrix_T_unlabel = cf_matrix(model_T, Dataset['v']['u']['sti'], Dataset['v']['u']['amp'], Dataset['v']['u']['target'])
        cf_matrix_S_unlabel = cf_matrix(model_S, Dataset['v']['u']['sti'], Dataset['v']['u']['amp'], Dataset['v']['u']['target'])
        
        # cf_matrix_T_unlabel = cf_matrix(model_T, Dataset['v']['u']['loader'])
        # cf_matrix_S_unlabel = cf_matrix(model_S, Dataset['v']['u']['loader'])
        # cf_matrix_test = cf_matrix(model_S, test_dataloader)
        
        # save_model(MODEL_PATH)
        save_plot(MODEL_PATH, EPOCHS, t_list, v_list)
        save_cm(MODEL_PATH, cf_matrix_T_unlabel, 'T_confusion_matrix')
        save_cm(MODEL_PATH, cf_matrix_S_unlabel, 'S_confusion_matrix')
#%% Load test data
        # DATA_PATH = 'dataset/20210604_nycu_toilet/csi.db'
        # scene_num = 1
        # timesteps = 50
        # valid_ratio = 0
        # test_data, _, test_target, _ = data_utils.load_data(DATA_PATH, scene_num, timesteps, valid_ratio)
        
        # test_sti = test_data[:,:,:224]
        # test_amp = test_data[:,:,224:]
        # del test_data
        
        # test_sti = torch.tensor(test_sti).type(torch.FloatTensor).to(device)
        # test_amp = torch.tensor(test_amp).type(torch.FloatTensor).to(device)
        # test_target = torch.tensor(test_target).type(torch.LongTensor).to(device)
        # cf_matrix_test = cf_matrix(model_S, test_sti, test_amp, test_target)
        # save_cm(MODEL_PATH, cf_matrix_test, 'test_confusion_matrix')
    except FileExistsError:
        print("Directory " , MODEL_PATH ,  " already exists")
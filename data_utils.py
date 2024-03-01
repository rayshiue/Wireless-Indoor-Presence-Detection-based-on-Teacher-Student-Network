# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:52:01 2021

@author: Mint
"""
import numpy as np
import math


def normalize_amp(amp,nFFT,nr,nc,offset):
    calibration_amp = np.zeros((len(amp), nFFT, nr*nc))
    for n in range(len(amp)):
        for rx in range(nr):
            for tx in range(nc):
                csi_amp_tmp = amp[n,:,rx*nc+tx]
                max_amp = max(csi_amp_tmp)
                min_amp = min(csi_amp_tmp)
                if np.isinf(min_amp):
                    min_amp = offset
                subst = csi_amp_tmp - min_amp
                if np.isinf(subst).any():
                    for s in range(subst.shape[0]):
                        if np.isinf(subst[s]):
                            subst[s] = offset
                calibration_amp[n,:,(rx)*nc+tx] = (subst) / (max_amp - min_amp) 
    return calibration_amp

def calibrate_phase(phase,nFFT,nr,nc):
    m = np.zeros(nFFT)
    for i in range(nFFT):
        m[i] = -28 + 1*(i+1)
    unwrapped_phase = np.zeros((len(phase), nFFT, nr*nc))
    calibration_phase = np.zeros((len(phase), nFFT, nr*nc))
    for n in range(len(phase)):
        for i in range(nr*nc):
            multi = 0
            sumTP = 0
            unwrapped_phase[n,0,i] = phase[n,0,i]
            for j in range(nFFT-1):
                diff = phase[n,j+1,i] - phase[n,j,i]
                if(diff) > math.pi:
                    multi = multi + 1
                elif(diff) < (-1)*math.pi:
                    multi = multi - 1
                unwrapped_phase[n,j+1,i] = phase[n,j+1,i] - multi*2*(math.pi)
                sumTP = sumTP + unwrapped_phase[n,j+1,i]
            k = (unwrapped_phase[n,nFFT-1,i]-unwrapped_phase[n,0,i]) / nFFT
            b = sumTP / nFFT     
            for j in range(nFFT):
                calibration_phase[n,j,i] = unwrapped_phase[n,j,i] - k*m[j] - b 
    return calibration_phase

def classify_data(case,y_test,calibration_amp,calibration_phase):
    calibrate_amp_list = []
    calibrate_phase_list = []
    for c in range(case):
        amp_temp = calibration_amp[(y_test[:]==c),:,:]
        calibrate_amp_list.append(amp_temp)
        phase_temp = calibration_phase[(y_test[:]==c),:,:]
        calibrate_phase_list.append(phase_temp)
    return calibrate_amp_list, calibrate_phase_list


def amp_STI(calibrate_amp_list,case,timesteps):
    for n in range(case):
        calibration_amp = calibrate_amp_list[n]
        calibration_amp = np.transpose(calibration_amp,(0,2,1))
        calibration_amp = np.reshape(calibration_amp, (calibration_amp.shape[0], calibration_amp.shape[1]*calibration_amp.shape[2]))
        calibration_amp_temp = np.zeros((calibration_amp.shape[0] - timesteps+1, timesteps, calibration_amp.shape[1]))
        for batch in range(calibration_amp.shape[0] - timesteps+1):
            calibration_amp_temp[batch,:,:] = calibration_amp[batch:batch+timesteps,:]
        sti = np.zeros((calibration_amp_temp.shape[0]-1, timesteps, calibration_amp_temp.shape[2]))
        for st in range(calibration_amp_temp.shape[0]-1):
            sti[st,:,:] = calibration_amp_temp[st+1,:,:] - calibration_amp_temp[st,:,:]
        if n == 0:
            sti_all = sti
            normali_amp = calibration_amp_temp[1:calibration_amp_temp.shape[0]]
        else:
            sti_all = np.row_stack((sti_all, sti))
            normali_amp = np.row_stack((normali_amp, calibration_amp_temp[1:calibration_amp_temp.shape[0]]))
    STI_amp = np.concatenate((sti_all,normali_amp),axis=2)
#    return sti_all
    return STI_amp


def phase_diff_time(calibrate_phase_list,case,timesteps):
    for c in range(case):
        phase_time_temp = np.reshape(calibrate_phase_list[c],(calibrate_phase_list[c].shape[0],calibrate_phase_list[c].shape[1]*calibrate_phase_list[c].shape[2]))
        phase_time = np.zeros((phase_time_temp.shape[0]-timesteps+1,timesteps,phase_time_temp.shape[1]))
        for batch in range(phase_time.shape[0]):
            phase_time[batch,:,:] = phase_time_temp[batch:batch+timesteps,:]
        phase_diffOfTime_tmp = np.zeros((phase_time.shape[0]-1,timesteps,phase_time.shape[2]))
        for d in range(phase_time.shape[0]-1):
            phase_diffOfTime_tmp[d,:,:] = phase_time[d+1,:,:] - phase_time[d,:,:]

        if c == 0:
            phase_diffOfTime = phase_diffOfTime_tmp
        else:
            phase_diffOfTime = np.row_stack((phase_diffOfTime, phase_diffOfTime_tmp))
    return phase_diffOfTime

def batch_size(case,timesteps,y_test):
    batch_size = 0
    batch_list = []
    for c in range(case):
        temp_size = y_test[(y_test[:]==c)].shape[0]
        temp_size = temp_size - timesteps
        batch_size = batch_size + temp_size
        batch_list.append(batch_size)
    return batch_list

def label_preprocess(y_test,case,timesteps):
    for k in range(case):
        y_test_tmp = y_test[(y_test==k)]
        y_test_tmp = y_test_tmp[timesteps:]
        if k == 0:
            label = y_test_tmp
        else:
            label = np.concatenate((label,y_test_tmp), axis=0)
    return label

def split_data(label,STI_amp,batch_list,new_case,timesteps, valid_ratio):
    # Merge data
    label = np.expand_dims(label, axis=1)
    label = np.expand_dims(label, axis=2)
    label = np.tile(label,(1,timesteps,1))
    
#    merge_data = np.concatenate((STI_amp, phase_diffOfTime), axis=2)
#    merge_data = np.concatenate((merge_data, label), axis=2)
    merge_data = np.concatenate((STI_amp, label), axis=2)
    
    for k in range(len(batch_list)):
        if k == 0:
            merge_data_temp = merge_data[0:batch_list[0], :, :]
            merge_data_batch = merge_data_temp[0:math.floor(merge_data_temp.shape[0]*(1-valid_ratio)), :, :]
            merge_data_valid = merge_data_temp[math.floor(merge_data_temp.shape[0]*(1-valid_ratio))+timesteps:, :, :]
        else:
            merge_data_temp = merge_data[batch_list[k-1]:batch_list[k], :, :]
            merge_data_batch = np.row_stack((merge_data_batch, merge_data_temp[0:math.floor(merge_data_temp.shape[0]*(1-valid_ratio)), :, :]))
            merge_data_valid = np.row_stack((merge_data_valid, merge_data_temp[math.floor(merge_data_temp.shape[0]*(1-valid_ratio))+timesteps:, :, :]))

    # Split data     
#    np.random.shuffle(merge_data_batch)
#    np.random.shuffle(merge_data_valid)
    input_dim = merge_data.shape[2] - 1
    X_train = merge_data_batch[0:merge_data_batch.shape[0], :, 0:input_dim]

#     # Get only few data
#     max_range = X_train.shape[0]
#     a_list = list(range(0, max_range))

#     np.random.shuffle(a_list)

# #    per = 1/32
# #    data_num = a_list[0:int(max_range*per)]
#     data_num = a_list[0:max_range]

#     selected = np.sort(data_num)

#     X_train = X_train[selected]

    y_train_temp = merge_data_batch[0:merge_data_batch.shape[0], 0, input_dim]
    # y_train_temp = y_train_temp[selected]
    
    X_valid = merge_data_valid[0:merge_data_valid.shape[0], :, 0:input_dim]
    y_valid_temp = merge_data_valid[0:merge_data_valid.shape[0], 0, input_dim]
    # Label
    # y_train = np.zeros((y_train_temp.shape[0], new_case))
    # for l in range(y_train_temp.shape[0]):
    #     y_train[l, int(y_train_temp[l])] = 1
    # y_valid = np.zeros((y_valid_temp.shape[0], new_case))
    # for l in range(y_valid_temp.shape[0]):
    #     y_valid[l, int(y_valid_temp[l])] = 1
    
    # y_train = y_train.astype(np.uint32)
    # y_valid = y_valid.astype(np.uint32)
    
    return X_train, X_valid, input_dim, y_train_temp, y_valid_temp

def get_condition(case,calibrate_amp_list,timesteps,selected, valid_ratio):
    for c in range(case):
        if c==0:
            normalized_amp_temp = calibrate_amp_list[c][timesteps:,:,:]
            normalized_amp = normalized_amp_temp[0:math.floor(normalized_amp_temp.shape[0]*(1-valid_ratio)), :, :]
            normalized_amp_valid = normalized_amp_temp[math.floor(normalized_amp_temp.shape[0]*(1-valid_ratio))+timesteps:, :, :]
        else:
            normalized_amp_temp = calibrate_amp_list[c][timesteps:,:,:]
            normalized_amp = np.row_stack((normalized_amp,normalized_amp_temp[0:math.floor(normalized_amp_temp.shape[0]*(1-valid_ratio)), :, :]))
            normalized_amp_valid = np.row_stack((normalized_amp_valid, normalized_amp_temp[math.floor(normalized_amp_temp.shape[0]*(1-valid_ratio))+timesteps:, :, :]))
    
    normalized_amp = normalized_amp[selected]
    normalized_amp = np.reshape(normalized_amp,(normalized_amp.shape[0],normalized_amp.shape[1]*normalized_amp.shape[2],1))
    normalized_amp_valid = np.reshape(normalized_amp_valid,(normalized_amp_valid.shape[0],normalized_amp_valid.shape[1]*normalized_amp_valid.shape[2],1))
    
    return normalized_amp, normalized_amp_valid

def split_unlabel(csi_data, csi_label, unlabel_percentage):
    '''
    This function is use to split the CSI data in each label to label data 
    and unlabel based on the unlabel percentage.

    Parameters
    ----------
    csi_data : numpy array
        CSI data that split to label data part and unlabel data part.
    csi_label : numpy array
        Labels of CSI data that split to label part and unlabel part.
    unlabel_percentage : float
        Float number range from 0~1.

    Returns
    -------
    label_data : numpy array
        Label data part.
    unlabel_data : numpy array
        Unlabel data part.
    label_target : numpy array
        Label part.
    unlabel_target : numpy array
        Unlabel part.

    '''
    start = 0
    end = 0
    classes = np.unique(csi_label)
    label_data_list = []
    unlabel_data_list = []
    label_list = []
    unlabel_list = []
    for i in range(len(classes)):
        total_len = sum(csi_label == classes[i])
        label_len = int(total_len * (1-unlabel_percentage))
        end += total_len
        label_data_list.append(csi_data[start:start + label_len])
        unlabel_data_list.append(csi_data[start + label_len:end])
        label_list.append(csi_label[start:start + label_len])
        unlabel_list.append(csi_label[start + label_len:end])
        start = end
    label_data = np.concatenate(label_data_list)
    unlabel_data = np.concatenate(unlabel_data_list)
    label_target = np.concatenate(label_list)
    unlabel_target = np.concatenate(unlabel_list)
    return label_data, unlabel_data, label_target, unlabel_target

def load_data(data_path, scene_num, timesteps, valid_ratio):
    import pickle
    np.random.seed(0)
    with open(data_path, 'rb') as file:
        scene = pickle.load(file)
    rx = scene[str(scene_num)]
    x_test = np.array(rx['x'])
    y_test = np.array(rx['y'])
    
    # Get phase and amplitude
    phase = np.angle(x_test)
    amp = abs(x_test)
    
    # Parameters
    offset = 1e-8
    nr = 2
    nc = 2
    case = 4
    new_case = 4
    nFFT = 56
    # eigenvector_list = []
    batch_list = []
    
    # Data preprocessing
    calibration_amp = normalize_amp(amp,nFFT,nr,nc,offset)
    del amp
    calibration_phase = calibrate_phase(phase,nFFT,nr,nc)
    del phase
    calibrate_amp_list, calibrate_phase_list = classify_data(case,y_test,calibration_amp,calibration_phase)
    del calibration_amp, calibration_phase
    STI_amp = amp_STI(calibrate_amp_list,case,timesteps)
    # phase_diffOfTime = phase_diff_time(calibrate_phase_list,case,timesteps)
    batch_list = batch_size(case,timesteps,y_test)
    label = label_preprocess(y_test,case,timesteps)
    del x_test, y_test
    train_data, valid_data, input_dim, train_target, valid_target = split_data(label,STI_amp,batch_list,new_case,timesteps, valid_ratio)
    del label, STI_amp
    # train_normalized_amp, valid_normalized_amp = get_condition(case,calibrate_amp_list,timesteps,selected, valid_ratio)
    # del calibrate_amp_list, selected
    return train_data, valid_data, train_target, valid_target



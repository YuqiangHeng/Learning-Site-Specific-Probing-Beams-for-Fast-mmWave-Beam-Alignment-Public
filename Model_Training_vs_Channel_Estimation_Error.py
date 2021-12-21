# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:34:01 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, fit
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook

np.random.seed(7)
# number of narrow data beams to select from (N_V)
n_nb = 128
# number of probing beams (N_W)
n_wb = 12
n_antenna = 64
antenna_sel = np.arange(n_antenna)
nepoch = 200
batch_size = 500    
noise_factor = -13 #dB
noise_power_dBm = -94
noiseless = False
    
h_est_NMSE_dB = np.arange(-34,-5,3)         
    
for dataset_name in ['Rosslyn_ULA','O1_28B_ULA','I3_60_ULA','O1_28_ULA']:
    # Training and testing data:
    if dataset_name == 'Rosslyn_ULA':
        h_real = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_real.npy')
        h_imag = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_imag.npy')
        tx_power_dBm = 10
    elif dataset_name == 'O1_28B_ULA':
        fname_h_real = './Dataset/DeepMIMO Dataset/O1_28B_ULA/h_real.mat'
        fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28B_ULA/h_imag.mat'
        h_real = sio.loadmat(fname_h_real)['h_real']
        h_imag = sio.loadmat(fname_h_imag)['h_imag']
        tx_power_dBm = 20
    elif dataset_name == 'I3_60_ULA':
        fname_h_real = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_real.mat'
        fname_h_imag = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_imag.mat'
        h_real = sio.loadmat(fname_h_real)['h_real']
        h_imag = sio.loadmat(fname_h_imag)['h_imag']
        tx_power_dBm = 10
    elif dataset_name == 'O1_28_ULA':
        fname_h_real = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_real.mat'
        fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_imag.mat'
        h_real = sio.loadmat(fname_h_real)['h_real']
        h_imag = sio.loadmat(fname_h_imag)['h_imag']
        tx_power_dBm = 10
    else:
        raise NameError('Dataset Not Supported')   
    
    if noiseless:
        noise_power_dBm = -np.inf    
    noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)   
        
    h = h_real + 1j*h_imag
    valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
    h = h[valid_ue_idc]
    h_real = h_real[valid_ue_idc]
    h_imag = h_imag[valid_ue_idc]
    norm_factor = np.max(abs(h))
    h_scaled = h/norm_factor
    h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)
    
    dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)

    train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
    val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
    
    # Data preparation
    label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
    soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)
    
    x_train,y_train = h_concat_scaled[train_idc,:],label[train_idc]
    x_val,y_val = h_concat_scaled[val_idc,:],label[val_idc]
    x_test,y_test = h_concat_scaled[test_idc,:],label[test_idc]
    
    x_train_sq_norm = np.linalg.norm(x_train,axis=1)**2
    
    torch_x_val,torch_y_val = torch.from_numpy(x_val),torch.from_numpy(y_val)
    torch_x_test,torch_y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
    # Pytorch train and test sets
    val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
    test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
    # data loader
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    learnable_codebook_acc_vs_NMSE = []
    learned_codebook_topk_gain_vs_NMSE = []
    learned_codebook_topk_acc_vs_NMSE = []
    
    for NMSE_dB_idx,NMSE_dB in enumerate(h_est_NMSE_dB):
        h_est_noise_pow = x_train_sq_norm.mean()*10**(NMSE_dB/10)
        x_train_noisy = x_train + np.random.normal(0,1,size=x_train.shape)*np.sqrt(h_est_noise_pow/2)
        
        h_train_noisy = x_train_noisy[:,:n_antenna] + 1j*x_train_noisy[:,n_antenna:]
        y_train_noisy = np.argmax(np.power(np.absolute(np.matmul(h_train_noisy, dft_nb_codebook.conj().T)),2),axis=1)
        
        torch_x_train,torch_y_train = torch.from_numpy(x_train_noisy),torch.from_numpy(y_train_noisy)
        train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
       
        learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                   trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
        learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        train_loss_hist, val_loss_hist = fit(learnable_codebook_model, train_loader, val_loader, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
        torch.save(learnable_codebook_model.state_dict(),'./Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_NMSE_{}dB_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,NMSE_dB,noise_power_dBm))
        plt.figure()
        plt.plot(train_loss_hist,label='training loss')
        plt.plot(val_loss_hist,label='validation loss')
        plt.legend()
        plt.title('{}: trainable codebook loss hist: {} wb {} nb NMSE = {} dB noise power = {} dBm'.format(dataset_name,n_wb,n_nb,NMSE_dB,noise_power_dBm))
        plt.show()
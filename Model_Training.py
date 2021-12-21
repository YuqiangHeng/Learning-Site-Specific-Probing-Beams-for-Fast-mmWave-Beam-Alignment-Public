# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, Beam_Classifier_CNN, fit, eval_model
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook

np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [6, 8, 10, 12, 14, 16, 18, 20]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [128 for i in n_wide_beams]
n_antenna = 64
nepoch = 200
batch_size = 500            
noise_factor = -13 #dB
noise_power_dBm = -94
noiseless = False

dataset_name = 'Rosslyn_ULA' # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
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

# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

torch_x_train = torch.from_numpy(x_train)
torch_y_train = torch.from_numpy(y_train)
torch_x_val = torch.from_numpy(x_val)
torch_y_val = torch.from_numpy(y_val)
torch_x_test = torch.from_numpy(x_test)
torch_y_test = torch.from_numpy(y_test)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)    

dft_codebooks = []
learned_codebooks = []
AMCF_codebooks = []

for n_wb_i, n_wb in enumerate(n_wide_beams):
    n_nb = n_narrow_beams[n_wb_i]
    print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
    dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
    label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
    soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

    x_train,y_train = h_concat_scaled[train_idc,:],label[train_idc]
    x_val,y_val = h_concat_scaled[val_idc,:],label[val_idc]
    x_test,y_test = h_concat_scaled[test_idc,:],label[test_idc]
    
    torch_x_train,torch_y_train = torch.from_numpy(x_train),torch.from_numpy(y_train)
    torch_x_val,torch_y_val = torch.from_numpy(x_val),torch.from_numpy(y_val)
    torch_x_test,torch_y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
    test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
    # Trainable probing codebook + MLP classifier, MLP takes the received power as input
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(learnable_codebook_model, train_loader, val_loader, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
    learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(learnable_codebook_model.state_dict(),learnable_model_savefname)
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('Trainable codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    learned_codebooks.append(learnable_codebook_model.get_codebook()) 
    
    # Evenly spaced DFT probing beams + MLP classifier, MLP takes the received power as input
    dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_codebook_opt = optim.Adam(dft_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(dft_codebook_model, train_loader, val_loader, dft_codebook_opt, nn.CrossEntropyLoss(), nepoch)
    dft_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(dft_codebook_model.state_dict(),dft_model_savefname)    
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('DFT codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    dft_codebooks.append(dft_codebook_model.get_codebook())

    # Wide AMCF probing beams + MLP classifier, MLP takes the received power as input    
    AMCF_wb_codebook = get_AMCF_codebook(n_wb, n_antenna)    
    AMCF_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                          trainable_codebook=False,complex_codebook=AMCF_wb_codebook,
                                          noise_power=noise_power,norm_factor=norm_factor)
    AMCF_codebook_opt = optim.Adam(AMCF_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(AMCF_codebook_model, train_loader, val_loader, AMCF_codebook_opt, nn.CrossEntropyLoss(), nepoch)
    AMCF_model_savefname = './Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(AMCF_codebook_model.state_dict(),AMCF_model_savefname)      
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('AMCF codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    AMCF_codebooks.append(AMCF_codebook_model.get_codebook())    

    # Evenly spaced DFT probing beams + CNN classifier, CNN takes the complex received signals as input
    dft_codebook_cnn_model = Beam_Classifier_CNN(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_codebook_cnn_opt = optim.Adam(dft_codebook_cnn_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(dft_codebook_cnn_model, train_loader, val_loader, dft_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
    dft_cnn_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(dft_codebook_cnn_model.state_dict(),dft_cnn_model_savefname)    
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('DFT codebook CNN loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()    
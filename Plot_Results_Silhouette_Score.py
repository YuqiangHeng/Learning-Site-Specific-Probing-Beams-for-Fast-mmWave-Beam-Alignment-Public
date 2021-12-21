# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:26:34 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from beam_utils import ULA_DFT_codebook as DFT_codebook

np.random.seed(7)
# number of narrow data beams to select from (N_V)
n_narrow_beams = 128
# number of probing beams (N_W)
n_wide_beams = [6, 8, 10, 12, 14, 16, 18, 20]
n_antenna = 64

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

h = h_real + 1j*h_imag
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
dft_nb_codebook = DFT_codebook(nseg=n_narrow_beams,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)

x_train,y_train = h_scaled[train_idc,:],label[train_idc]
x_val,y_val = h_scaled[val_idc,:],label[val_idc]
x_test,y_test = h_scaled[test_idc,:],label[test_idc]

print(dataset_name)
for codebook_type in ['trainable','AMCF','DFT']:
    if codebook_type == 'trainable':
        print_prefix = '& learned'
    else:
        print_prefix = '& '+codebook_type
    s_scores = []
    for i,N in enumerate(n_wide_beams):             
        probing_codebook = np.load('./Saved Codebooks/{}_probe_{}_codebook_{}_beam.npy'.format(dataset_name,codebook_type,N))
        h_project_probing = np.power(np.absolute(np.matmul(x_test, probing_codebook.conj().T)),2)
        probing_silhouette_score = silhouette_score(h_project_probing, y_test)
        s_scores.append(probing_silhouette_score)
    s_scores_char = print_prefix
    for i in s_scores:
        s_scores_char += ' & {:.3f}'.format(i)
    if codebook_type == 'DFT':
        s_scores_char += ' \\\\ \\hline'
    else:
        s_scores_char += ' \\\\ \\cline{2-10}'
    print(s_scores_char)

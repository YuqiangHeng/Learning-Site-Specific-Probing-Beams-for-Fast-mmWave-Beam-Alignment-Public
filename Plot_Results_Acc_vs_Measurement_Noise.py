# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:01:23 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier
import torch.utils.data
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import DFT_angles, Beam_Search_Tree, AMCF_boundaries, get_AMCF_codebook

np.random.seed(7)

# number of probing beams (N_W)
n_wb = 12
# number of narrow data beams to select from (N_V)
n_nb = 128

n_antenna = 64
batch_size = 500
noise_factor = -13 #dB

model_noise_power_dBm = -94
noiseless = False

target_avg_snr_dB = np.arange(-10,50,5)

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
valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
h = h[valid_ue_idc]
h_real = h_real[valid_ue_idc]
h_imag = h_imag[valid_ue_idc]
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)

dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
avg_opti_bf_gain_dB = 10*np.log10(np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2).max(axis=1).mean())

noise_power_dBm_array = tx_power_dBm + avg_opti_bf_gain_dB - target_avg_snr_dB
noise_power_dBm_array = np.insert(noise_power_dBm_array,0,-np.inf)
noise_power_array = 10**((noise_power_dBm_array-tx_power_dBm)/10)
            
learnable_codebook_topk_acc_vs_noise = []

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

for noise_power_dBm, noise_power in zip(noise_power_dBm_array,noise_power_array):
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                               trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_codebook_model.load_state_dict(torch.load('./Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)))
    
    y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
    topk_bf_gain_learnable_codebook = []
    topk_acc_learnable_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_learned_codebook):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_learnable_codebook.append(topk_acc)
    topk_acc_learnable_codebook = np.array(topk_acc_learnable_codebook).mean(axis=0)
    learnable_codebook_topk_acc_vs_noise.append(topk_acc_learnable_codebook)

"""    
compute exhaustive and 2-iter hierarchical beam search acc
"""
exhaustive_acc_vs_noise = []
dft_nb_codebook = DFT_codebook(nseg=128,n_antenna=64)
dft_nb_az = DFT_angles(128)
dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
best_nb = np.argmax(nb_bf_gain,axis=1)
best_nb_az = dft_nb_az[best_nb]
    
two_tier_AMCF_acc_vs_noise = []

AMCF_wb_codebook = get_AMCF_codebook(n_wb, n_antenna)
AMCF_wb_cv = AMCF_boundaries(n_wb)
#handle numerical issue where theta is not exactly +-1
AMCF_wb_cv[AMCF_wb_cv>1.0]=1.0
AMCF_wb_cv[AMCF_wb_cv<-1.0]=-1.0
AMCF_wb_cv = np.arcsin(AMCF_wb_cv)
AMCF_wb_cv = np.flipud(AMCF_wb_cv)

wb_2_nb = {}
for bi in range(n_wb):
    children_nb = ((dft_nb_az>=AMCF_wb_cv[bi,0]) & (dft_nb_az<=AMCF_wb_cv[bi,1])).nonzero()[0]
    wb_2_nb[bi] = children_nb
        
wb_bf_signal = np.matmul(h[test_idc], AMCF_wb_codebook.conj().T)
wb_bf_gain = np.power(np.absolute(wb_bf_signal),2)

for noise_power in noise_power_array:
    nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
    nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
    nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
    nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
    best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
    exhaustive_acc = (best_nb_noisy==best_nb).mean()
    exhaustive_acc_vs_noise.append(exhaustive_acc)
    
    wb_bf_noise_real = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_signal_with_noise = wb_bf_signal + wb_bf_noise_real + 1j*wb_bf_noise_imag
    wb_bf_gain_with_noise = np.power(np.absolute(wb_bf_signal_with_noise),2)
    best_wb_noisy = np.argmax(wb_bf_gain_with_noise,axis=1)

    two_tier_best_nb = []
    for ue_idx,best_wb_idx in enumerate(best_wb_noisy):
        child_beam_idc = wb_2_nb[best_wb_idx]
        child_beam_gain_noisy = nb_bf_gain_with_noise[ue_idx,child_beam_idc]
        two_tier_best_nb.append(child_beam_idc[np.argmax(child_beam_gain_noisy)])
    two_tier_best_nb = np.array(two_tier_best_nb)    
    hierarchical_acc = (two_tier_best_nb==best_nb).mean() 
    two_tier_AMCF_acc_vs_noise.append(hierarchical_acc)

exhaustive_acc_vs_noise = np.array(exhaustive_acc_vs_noise)    
two_tier_AMCF_acc_vs_noise = np.array(two_tier_AMCF_acc_vs_noise)

"""
binary beam search using AMCF wide beams
"""
bst_acc_vs_noise = []
bst_true_snr_vs_noise = []
for noise_power in noise_power_array:
    bst = Beam_Search_Tree(n_antenna=n_antenna,n_narrow_beam=128,k=2,noise_power=noise_power)
    bst_bf_gain, bst_nb_idx = bst.forward_batch(h[test_idc])
    bst_acc = (bst_nb_idx==best_nb).mean()
    bst_acc_vs_noise.append(bst_acc)
bst_acc_vs_noise = np.array(bst_acc_vs_noise)
    
"""
plotting
"""    

plt.figure(figsize=(6,4.5))
plt.plot(target_avg_snr_dB,exhaustive_acc_vs_noise[0]-exhaustive_acc_vs_noise[1:],linestyle='solid',marker='o',label='Exhaustive search')
plt.plot(target_avg_snr_dB,learnable_codebook_topk_acc_vs_noise[0,2]-learnable_codebook_topk_acc_vs_noise[1:,2],'--',marker='s',label='Proposed method, k={}'.format(3))
plt.plot(target_avg_snr_dB,learnable_codebook_topk_acc_vs_noise[0,0]-learnable_codebook_topk_acc_vs_noise[1:,0],marker='s',label='Proposed method, k={}'.format(1))
plt.plot(target_avg_snr_dB,two_tier_AMCF_acc_vs_noise[0]-two_tier_AMCF_acc_vs_noise[1:],marker='x',label='2-tier hierarchical search')
plt.plot(target_avg_snr_dB,bst_acc_vs_noise[0]-bst_acc_vs_noise[1:],linestyle='dotted',marker='+',label='Binary search')
plt.xticks(target_avg_snr_dB)
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('$\Delta$ accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()
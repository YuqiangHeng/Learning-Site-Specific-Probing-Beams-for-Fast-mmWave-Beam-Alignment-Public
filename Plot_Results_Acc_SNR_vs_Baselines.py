# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:04:13 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, eval_model
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import plot_codebook_pattern, DFT_angles, Beam_Search_Tree, get_AMCF_codebook, AMCF_boundaries

np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [6, 8, 10, 12, 14, 16, 18, 20]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [128 for i in n_wide_beams]
n_antenna = 64
batch_size = 500

noise_power_dBm = -94
noise_factor = -13 #dB

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

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)

dft_codebook_acc = []
learnable_codebook_acc = []
AMCF_codebook_acc = []

dft_codebook_topk_acc = []
learnable_codebook_topk_acc = []
AMCF_codebook_topk_acc = []

dft_codebook_topk_gain = []
learned_codebook_topk_gain= []
AMCF_codebook_topk_gain = []

optimal_gains = []
learned_codebooks = []
dft_codebooks = []
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
    
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                               trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    learnable_codebook_model.load_state_dict(torch.load(learnable_model_savefname))
    learnable_codebook_test_loss,learnable_codebook_test_acc = eval_model(learnable_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    learnable_codebook_acc.append(learnable_codebook_test_acc)
    y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
    topk_bf_gain_learnable_codebook = []
    topk_acc_learnable_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_learned_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_learnable_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_learnable_codebook.append(topk_acc)
    topk_bf_gain_learnable_codebook = np.array(topk_bf_gain_learnable_codebook)
    learned_codebook_topk_gain.append(topk_bf_gain_learnable_codebook)
    learned_codebooks.append(learnable_codebook_model.get_codebook()) 
    topk_acc_learnable_codebook = np.array(topk_acc_learnable_codebook).mean(axis=0)
    learnable_codebook_topk_acc.append(topk_acc_learnable_codebook)

    
    dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    dft_codebook_model.load_state_dict(torch.load(dft_model_savefname))
    dft_codebook_test_loss,dft_codebook_test_acc = eval_model(dft_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    dft_codebook_acc.append(dft_codebook_test_acc)
    y_test_predict_dft_codebook = dft_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
    topk_bf_gain_dft_codebook = []
    topk_acc_dft_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_dft_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_dft_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_dft_codebook.append(topk_acc)        
    topk_bf_gain_dft_codebook = np.array(topk_bf_gain_dft_codebook)
    dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
    dft_codebooks.append(dft_codebook_model.get_codebook())
    topk_acc_dft_codebook = np.array(topk_acc_dft_codebook).mean(axis=0)
    dft_codebook_topk_acc.append(topk_acc_dft_codebook)
    
    AMCF_wb_codebook = get_AMCF_codebook(n_wb, n_antenna)
    AMCF_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                          trainable_codebook=False,complex_codebook=AMCF_wb_codebook,
                                          noise_power=noise_power,norm_factor=norm_factor)
    AMCF_model_savefname = './Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    AMCF_codebook_model.load_state_dict(torch.load(AMCF_model_savefname))    
    AMCF_codebook_test_loss,AMCF_codebook_test_acc = eval_model(AMCF_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    AMCF_codebook_acc.append(AMCF_codebook_test_acc)
    y_test_predict_AMCF_codebook = AMCF_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_AMCF_codebook = (-y_test_predict_AMCF_codebook).argsort()
    topk_bf_gain_AMCF_codebook = []
    topk_acc_AMCF_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_AMCF_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_AMCF_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_AMCF_codebook.append(topk_acc)        
    topk_bf_gain_AMCF_codebook = np.array(topk_bf_gain_AMCF_codebook)
    AMCF_codebook_topk_gain.append(topk_bf_gain_AMCF_codebook)
    AMCF_codebooks.append(AMCF_codebook_model.get_codebook())
    topk_acc_AMCF_codebook = np.array(topk_acc_AMCF_codebook).mean(axis=0)
    AMCF_codebook_topk_acc.append(topk_acc_AMCF_codebook)
    
    optimal_gains.append(soft_label[test_idc,:].max(axis=-1))
 
dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
AMCF_codebook_topk_gain = np.array(AMCF_codebook_topk_gain)

optimal_gains = np.array(optimal_gains)

dft_codebook_topk_snr = tx_power_dBm + 10*np.log10(dft_codebook_topk_gain) - noise_power_dBm + noise_factor
learned_codebook_topk_snr = tx_power_dBm + 10*np.log10(learned_codebook_topk_gain) - noise_power_dBm + noise_factor
AMCF_codebook_topk_snr = tx_power_dBm + 10*np.log10(AMCF_codebook_topk_gain) - noise_power_dBm + noise_factor
optimal_snr = tx_power_dBm + 10*np.log10(optimal_gains) - noise_power_dBm + noise_factor


dft_codebook_topk_acc = np.array(dft_codebook_topk_acc)
learnable_codebook_topk_acc = np.array(learnable_codebook_topk_acc)
AMCF_codebook_topk_acc = np.array(AMCF_codebook_topk_acc)

"""    
compute exhaustive beam search acc and snr
"""
dft_nb_codebook = DFT_codebook(nseg=128,n_antenna=64)
dft_nb_az = DFT_angles(128)
dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
best_nb = np.argmax(nb_bf_gain,axis=1)
best_nb_az = dft_nb_az[best_nb]
exhaustive_acc = (best_nb_noisy==best_nb).mean()
genie_nb_snr = tx_power_dBm + 10*np.log10(nb_bf_gain.max(axis=1)) - noise_power_dBm + noise_factor
exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
exhaustive_nb_snr = tx_power_dBm + 10*np.log10(exhaustive_nb_snr) - noise_power_dBm + noise_factor
print('Genie avg. SNR = {} dB.'.format(genie_nb_snr.mean()))
print('Exhaustive search accuracy = {}, avg. SNR = {} dB.'.format(exhaustive_acc,exhaustive_nb_snr.mean()))

"""    
compute 2-iter hierarchical beam search acc and snr
"""    
two_tier_AMCF_acc = []
two_tier_AMCF_snr = []
for wb_i, n_wb in enumerate(n_wide_beams):
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
    wb_bf_noise_real = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_signal_with_noise = wb_bf_signal + wb_bf_noise_real + 1j*wb_bf_noise_imag
    wb_bf_gain = np.power(np.absolute(wb_bf_signal),2)
    wb_bf_gain_with_noise = np.power(np.absolute(wb_bf_signal_with_noise),2)
    best_wb_noisy = np.argmax(wb_bf_gain_with_noise,axis=1)
    
    two_tier_best_nb = []
    for ue_idx,best_wb_idx in enumerate(best_wb_noisy):
        child_beam_idc = wb_2_nb[best_wb_idx]
        child_beam_gain_noisy = nb_bf_gain_with_noise[ue_idx,child_beam_idc]
        two_tier_best_nb.append(child_beam_idc[np.argmax(child_beam_gain_noisy)])
    two_tier_best_nb = np.array(two_tier_best_nb)    
    hierarchical_acc = (two_tier_best_nb==best_nb).mean() 
    
    two_tier_AMCF_acc.append(hierarchical_acc)
    
    best_wb_best_child_nb_snr = np.array([nb_bf_gain[ue_idx,two_tier_best_nb_idx] for ue_idx,two_tier_best_nb_idx in enumerate(two_tier_best_nb)])
    best_wb_best_child_nb_snr = tx_power_dBm + 10*np.log10(best_wb_best_child_nb_snr) - noise_power_dBm + noise_factor
    two_tier_AMCF_snr.append(best_wb_best_child_nb_snr)
two_tier_AMCF_acc = np.array(two_tier_AMCF_acc)
two_tier_AMCF_snr = np.array(two_tier_AMCF_snr)

print('2-tier hierarchical search accuracy:')
print(two_tier_AMCF_acc)
print('2-tier hierarchical search avg. SNR:')
print(two_tier_AMCF_snr.mean(axis=1))

"""
binary beam search using AMCF wide beams
"""

bst = Beam_Search_Tree(n_antenna=n_antenna,n_narrow_beam=n_narrow_beams[0],k=2,noise_power=noise_power)
bst_bf_gain, bst_nb_idx = bst.forward_batch(h[test_idc])
bst_acc = (bst_nb_idx==best_nb).mean()
bst_true_snr = nb_bf_gain[tuple(np.arange(nb_bf_gain.shape[0])),tuple(bst_nb_idx)]
bst_true_snr = tx_power_dBm + 10*np.log10(bst_true_snr) - noise_power_dBm + noise_factor
print('BST acc = {}, avg. SNR = {}'.format(bst_acc, bst_true_snr.mean()))

plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learnable_codebook_acc,marker='s',label='Learned probing codebook')
plt.plot(n_wide_beams,dft_codebook_acc,marker='+',label='DFT probing codebook')
plt.plot(n_wide_beams,AMCF_codebook_acc,marker='o',label='AMCF probing codebook')
plt.legend()
plt.xticks(n_wide_beams)
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Acc vs. Probing Codebook')
plt.show()

plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,2],'--',marker='s',label='Proposed method, k={}'.format(3))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,1],'-.',marker='s',label='Proposed method, k={}'.format(2))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,0],marker='s',label='Proposed method, k={}'.format(1))
plt.plot(n_wide_beams,two_tier_AMCF_acc,marker='x',label='2-tier hierarchical search')
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.hlines(y=bst_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='Proposed method, k={}'.format(3))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='Proposed method, k={}'.format(2))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='Proposed method, k={}'.format(1))
plt.plot(n_wide_beams,two_tier_AMCF_snr.mean(axis=1),marker='x',label='2-tier hierarchical search')
plt.hlines(y=genie_nb_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='solid',label='Genie')
plt.hlines(y=exhaustive_nb_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.hlines(y=bst_true_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Average SNR (dB)')
plt.title('SNR of top-k predicted beams')
plt.show()

for i,N in enumerate(n_wide_beams):     
    fig,ax = plot_codebook_pattern(learned_codebooks[i].T)
    ax.set_title('Trainable {}-Beam Codebook'.format(N))
    fig,ax = plot_codebook_pattern(dft_codebooks[i])
    ax.set_title('DFT {}-Beam Codebook'.format(N))
    fig,ax = plot_codebook_pattern(AMCF_codebooks[i])
    ax.set_title('AMCF {}-Beam Codebook'.format(N))

for i,N in enumerate(n_wide_beams):  
    np.save('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,N),learned_codebooks[i].T)
    np.save('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,N),dft_codebooks[i])    
    np.save('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,N),AMCF_codebooks[i])   

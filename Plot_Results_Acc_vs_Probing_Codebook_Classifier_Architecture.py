# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:49:50 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, Beam_Classifier_CNN, Beam_Classifier_CNN_RSRP
import torch.utils.data
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import DFT_angles, Beam_Search_Tree, AMCF_boundaries, get_AMCF_codebook

np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [6, 8, 10, 12, 14, 16, 18, 20]
# number of narrow data beams to select from (N_V)
n_nb = 128
n_antenna = 64
batch_size = 500

noise_power_dBm = -94
noise_factor = -13 #dB

noiseless_model = False

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

if noiseless_model:
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

learned_codebook_mlp_topk_acc = []
learned_codebook_cnn_topk_acc = []
dft_codebook_cnn_topk_acc = []
dft_codebook_mlp_topk_acc = []
AMCF_codebook_mlp_topk_acc = []

for n_wb_i, n_wb in enumerate(n_wide_beams):
    print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))   
    """
    Trainable codebook with MLP classifier
    """
    trainable_codebook_mlp_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                               trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    trainable_codebook_mlp_model.load_state_dict(torch.load(learnable_model_savefname))
    y_test_predict = trainable_codebook_mlp_model(torch_x_test.float()).detach().numpy()
    topk_acc_test = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],(-y_test_predict).argsort()):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_test.append(topk_acc)
    learned_codebook_mlp_topk_acc.append(np.array(topk_acc_test).mean(axis=0))

    """
    Trainable codebook with CNN classifier
    """
    trainable_codebook_cnn_model = Beam_Classifier_CNN_RSRP(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                               trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_CNN_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    trainable_codebook_cnn_model.load_state_dict(torch.load(learnable_model_savefname))
    y_test_predict = trainable_codebook_cnn_model(torch_x_test.float()).detach().numpy()
    topk_acc_test = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],(-y_test_predict).argsort()):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_test.append(topk_acc)
    learned_codebook_cnn_topk_acc.append(np.array(topk_acc_test).mean(axis=0))

    """
    DFT codebook with CNN classifier (complex input)
    """    
    dft_codebook_cnn_model = Beam_Classifier_CNN(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=DFT_codebook(nseg=n_wb,n_antenna=n_antenna),
                                         noise_power=noise_power,norm_factor=norm_factor)
    learnable_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    dft_codebook_cnn_model.load_state_dict(torch.load(learnable_model_savefname))
    y_test_predict = dft_codebook_cnn_model(torch_x_test.float()).detach().numpy()
    topk_acc_test = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],(-y_test_predict).argsort()):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_test.append(topk_acc)
    dft_codebook_cnn_topk_acc.append(np.array(topk_acc_test).mean(axis=0))    

    """
    DFT codebook with MLP classifier
    """   
    dft_codebook_mlp_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=DFT_codebook(nseg=n_wb,n_antenna=n_antenna),
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    dft_codebook_mlp_model.load_state_dict(torch.load(dft_model_savefname))
    y_test_predict = dft_codebook_mlp_model(torch_x_test.float()).detach().numpy()
    topk_acc_test = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],(-y_test_predict).argsort()):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_test.append(topk_acc)        
    topk_acc_test = np.array(topk_acc_test).mean(axis=0)
    dft_codebook_mlp_topk_acc.append(topk_acc_test)

    """
    AMCF codebook with MLP classifier
    """       
    AMCF_wb_codebook = get_AMCF_codebook(n_wb, n_antenna)    
    AMCF_codebook_mlp_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                          trainable_codebook=False,complex_codebook=AMCF_wb_codebook,
                                          noise_power=noise_power,norm_factor=norm_factor)
    AMCF_model_savefname = './Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    AMCF_codebook_mlp_model.load_state_dict(torch.load(AMCF_model_savefname))    
    y_test_predict = AMCF_codebook_mlp_model(torch_x_test.float()).detach().numpy()
    topk_acc_test = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],(-y_test_predict).argsort()):
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_test.append(topk_acc)        
    topk_acc_test = np.array(topk_acc_test).mean(axis=0)
    AMCF_codebook_mlp_topk_acc.append(topk_acc_test)

learned_codebook_mlp_topk_acc = np.array(learned_codebook_mlp_topk_acc)
learned_codebook_cnn_topk_acc = np.array(learned_codebook_cnn_topk_acc)
dft_codebook_cnn_topk_acc = np.array(dft_codebook_cnn_topk_acc)
dft_codebook_mlp_topk_acc = np.array(dft_codebook_mlp_topk_acc)
AMCF_codebook_mlp_topk_acc = np.array(AMCF_codebook_mlp_topk_acc)


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
two_tier_AMCF_acc = np.array(two_tier_AMCF_acc)

"""
binary beam search using AMCF wide beams
"""

bst = Beam_Search_Tree(n_antenna=n_antenna,n_narrow_beam=n_nb,k=2,noise_power=noise_power)
bst_bf_gain, bst_nb_idx = bst.forward_batch(h[test_idc])
bst_acc = (bst_nb_idx==best_nb).mean()


"""
Statistical codebook using top N_wb most used narrow beams
"""
y_train_noisy = np.matmul(h[train_idc], DFT_codebook(nseg=n_nb,n_antenna=n_antenna).conj().T)
y_train_noisy = y_train_noisy + np.random.normal(loc=0,scale=1,size=y_train_noisy.shape)*np.sqrt(noise_power/2) + 1j*np.random.normal(loc=0,scale=1,size=y_train_noisy.shape)*np.sqrt(noise_power/2)
y_train_noisy = np.argmax(np.power(np.absolute(y_train_noisy),2),axis=1)
unique_beams, beam_counts = np.unique(y_train_noisy, return_counts=True)
beam_sorted_by_freq = unique_beams[(-beam_counts).argsort()]

rsrp_test_noisy = np.matmul(h[test_idc], DFT_codebook(nseg=n_nb,n_antenna=n_antenna).conj().T)
rsrp_test_noisy = rsrp_test_noisy + np.random.normal(loc=0,scale=1,size=rsrp_test_noisy.shape)*np.sqrt(noise_power/2) + 1j*np.random.normal(loc=0,scale=1,size=rsrp_test_noisy.shape)*np.sqrt(noise_power/2)
rsrp_test_noisy = np.power(np.absolute(rsrp_test_noisy),2)

statistical_codebook_bf_gain = []
for n_wb in n_wide_beams:
    rsrp_test_noisy_statistical_codebook = rsrp_test_noisy[:,beam_sorted_by_freq[:n_wb]]
    y_test_pred_statistical_codebook = rsrp_test_noisy_statistical_codebook.argmax(axis=1)
    y_test_pred_statistical_codebook = beam_sorted_by_freq[:n_wb][y_test_pred_statistical_codebook]
    statistical_codebook_bf_gain.append(soft_label[test_idc,:][np.arange(len(test_idc)),y_test_pred_statistical_codebook])
statistical_codebook_acc = np.array([np.in1d(y_test,beam_sorted_by_freq[:n_wb]).sum()/len(y_test) for n_wb in n_wide_beams])
    
"""
plotting
"""    

plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learned_codebook_mlp_topk_acc[:,0],marker='s',label='Proposed: learned codebook + MLP')
plt.plot(n_wide_beams,learned_codebook_cnn_topk_acc[:,0],marker='+',label='Learned codebook + CNN')
plt.plot(n_wide_beams,AMCF_codebook_mlp_topk_acc[:,0],linestyle='dotted',marker='^',label='AMCF codebook + MLP')
plt.plot(n_wide_beams,dft_codebook_mlp_topk_acc[:,0],marker='o',label='DFT codebook + MLP')
plt.plot(n_wide_beams,dft_codebook_cnn_topk_acc[:,0],'--',marker='x',label='AMPBML: DFT codebook + CNN')
plt.xticks(n_wide_beams)
plt.yticks(np.arange(0.2,1,0.2))
plt.ylim(top=1)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
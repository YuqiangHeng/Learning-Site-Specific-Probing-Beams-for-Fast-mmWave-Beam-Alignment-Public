# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:04:53 2021

@author: Yuqiang (Ethan) Heng
"""

import time
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
import seaborn as sns

np.random.seed(7)
# number of narrow data beams to select from (N_V)
n_nb = 128
# number of probing beams (N_W)
n_wide_beams = [16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
tsne_perplexity = 20

dataset_name = 'Rosslyn_ULA' # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'

# Training and testing data:
if dataset_name == 'Rosslyn_ULA':
    h_real = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_real.npy')
    h_imag = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_imag.npy')
elif dataset_name == 'O1_28B_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/O1_28B_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28B_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
elif dataset_name == 'I3_60_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
elif dataset_name == 'O1_28_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
else:
    raise NameError('Dataset Not Supported')

h = h_real + 1j*h_imag
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.2)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)


x_train,y_train = h_scaled[train_idc,:],label[train_idc]
x_test,y_test = h_scaled[test_idc,:],label[test_idc]
    
for i,N in enumerate(n_wide_beams):  

    trainable_codebook = np.load('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,N))
    dft_codebook = np.load('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,N))
    AMCF_codebook = np.load('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,N))
    
    feat_cols = ['beam_{}'.format(bi) for bi in range(N)]
    h_project_trainable = np.power(np.absolute(np.matmul(x_test, trainable_codebook.conj().T)),2)
    df_trainable = pd.DataFrame(h_project_trainable,columns=feat_cols)
    df_trainable['y'] = y_test
    df_trainable['label'] = df_trainable['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_trainable = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    trainable_tsne_results = tsne_trainable.fit_transform(df_trainable[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_trainable['t-SNE Dim 1'] = trainable_tsne_results[:,0]
    df_trainable['t-SNE Dim 2'] = trainable_tsne_results[:,1]
    

    h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
    df_dft = pd.DataFrame(h_project_dft,columns=feat_cols)
    df_dft['y'] = y_test
    df_dft['label'] = df_dft['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_dft = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    dft_tsne_results = tsne_dft.fit_transform(df_dft[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_dft['t-SNE Dim 1'] = dft_tsne_results[:,0]
    df_dft['t-SNE Dim 2'] = dft_tsne_results[:,1]    
    
    h_project_AMCF = np.power(np.absolute(np.matmul(x_test, AMCF_codebook.conj().T)),2)
    df_AMCF = pd.DataFrame(h_project_AMCF,columns=feat_cols)
    df_AMCF['y'] = y_test
    df_AMCF['label'] = df_AMCF['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_AMCF = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    AMCF_tsne_results = tsne_AMCF.fit_transform(df_AMCF[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_AMCF['t-SNE Dim 1'] = AMCF_tsne_results[:,0]
    df_AMCF['t-SNE Dim 2'] = AMCF_tsne_results[:,1] 

    fig1,ax1 = plt.subplots(figsize=(4.5,4.5))
    sns.scatterplot(
        x='t-SNE Dim 1', y='t-SNE Dim 2',
        hue="y",
        palette=sns.color_palette("hls", len(df_trainable['y'].unique())),
        data=df_trainable,
        legend=False,
        alpha=0.3,
        ax=ax1
    )    
    ax1.set_xticks(np.arange(-150,200,50))
    ax1.set_yticks(np.arange(-150,200,50))
    
    fig2,ax2 = plt.subplots(figsize=(4.5,4.5))
    sns.scatterplot(
        x='t-SNE Dim 1', y='t-SNE Dim 2',
        hue="y",
        palette=sns.color_palette("hls", len(df_dft['y'].unique())),
        data=df_dft,
        legend=False,
        alpha=0.3,
        ax=ax2
    )   
    ax2.set_xticks(np.arange(-150,200,50))
    ax2.set_yticks(np.arange(-150,200,50))

    fig3,ax3 = plt.subplots(figsize=(4.5,4.5))
    sns.scatterplot(
        x='t-SNE Dim 1', y='t-SNE Dim 2',
        hue="y",
        palette=sns.color_palette("hls", len(df_AMCF['y'].unique())),
        data=df_AMCF,
        legend=False,
        alpha=0.3,
        ax=ax3
    )   
    ax3.set_xticks(np.arange(-150,200,50))
    ax3.set_yticks(np.arange(-150,200,50))
    

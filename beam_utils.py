# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:23:40 2021

@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def DFT_angles(n_beam):
    delta_theta = 1/n_beam
    if n_beam % 2 == 1:
        thetas = np.arange(0,1/2,delta_theta)
        # thetas = np.linspace(0,1/2,n_beam//2+1,endpoint=False)
        thetas = np.concatenate((-np.flip(thetas[1:]),thetas))
    else:
        thetas = np.arange(delta_theta/2,1/2,delta_theta) 
        thetas = np.concatenate((-np.flip(thetas),thetas))
    return thetas

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def DFT_beam(n_antenna,azimuths):
    codebook_all = np.zeros((len(azimuths),n_antenna),dtype=np.complex_)
    for i,phi in enumerate(azimuths):
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def DFT_beam_blockmatrix(n_antenna,azimuths):
    codebook = DFT_beam(n_antenna,azimuths).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def ULA_DFT_codebook_blockmatrix(nseg,n_antenna):
    codebook = ULA_DFT_codebook(nseg,n_antenna).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def codebook_blockmatrix(codebook):
    # codebook has dimension n_antenna x n_beams
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def bf_gain_loss(y_pred, y_true):
    return -torch.mean(y_pred,dim=0)

def calc_beam_pattern(beam, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
    phi_all = np.linspace(-np.pi/2,np.pi/2,resolution)
    array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
    array_response_vectors = -1j*2*np.pi*k*np.sin(array_response_vectors)
    array_response_vectors = array_response_vectors * np.arange(n_antenna)
    array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
    gains = abs(array_response_vectors.conj() @ beam)**2
    return phi_all, gains

def plot_codebook_pattern(codebook):
    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_beam_pattern(beam)
        ax.plot(phi,bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    return fig, ax
    
def plot_codebook_pattern_on_axe(codebook,ax):
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_beam_pattern(beam)
        ax.plot(phi,bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    
    
def get_AMCF_beam(omega_min, omega_max, n_antenna=64,spacing=0.5,Q=2**10): 
    #omega_min and omega_max are the min and max beam coverage, Q is the resolution
    q = np.arange(Q)+1
    omega_q = -1 + (2*q-1)/Q
    #array response matrix
    A_phase = np.outer(np.arange(n_antenna),omega_q)
    A = np.exp(1j*np.pi*A_phase)
    #physical angles between +- 90 degree
    theta_min, theta_max = np.arcsin(omega_min),np.arcsin(omega_max)
    #beamwidth in spatial angle
    B = omega_max - omega_min
    mainlobe_idc = ((omega_q >= omega_min) & (omega_q <= omega_max)).nonzero()[0]
    sidelobe_idc = ((omega_q < omega_min) | (omega_q > omega_max)).nonzero()[0]
    #ideal beam amplitude pattern
    g = np.zeros(Q)
    g[mainlobe_idc] = np.sqrt(2/B)
    #g_eps = g in mainlobe and = eps in sidelobe to avoid log0=Nan
    eps = 2**(-52)
    g_eps = g
    g_eps[sidelobe_idc] = eps
    
    v0_phase = B*np.arange(n_antenna)*np.arange(1,n_antenna+1)/2/n_antenna + np.arange(n_antenna)*omega_min
    v0 = 1/np.sqrt(n_antenna)*np.exp(1j*np.pi*v0_phase.conj().T)
    v = v0
    ite = 1
    mse_history = []
    while True:
        mse = np.power(abs(A.conj().T @ v) - g,2).mean()
        mse_history.append(mse)
        if ite >= 10 and abs(mse_history[-1] - mse_history[-2]) < 0.01*mse_history[-1]:
            break
        else:
            ite += 1
        Theta = np.angle(A.conj().T @ v)
        r = g * np.exp(1j*Theta)
        v = 1/np.sqrt(n_antenna)*np.exp(1j*np.angle(A @ r))
    return v
        
def AMCF_boundaries(n_beams):
    beam_boundaries = np.zeros((n_beams,2))
    for k in range(n_beams):
        beam_boundaries[k,0] = -1 + k*2/n_beams
        beam_boundaries[k,1] = beam_boundaries[k,0] + 2/n_beams
    return beam_boundaries

def get_AMCF_codebook(n_beams,n_antenna,spacing=0.5):
    AMCF_codebook_all = np.zeros((n_beams,n_antenna),dtype=np.complex_)
    AMCF_boundary = AMCF_boundaries(n_beams)
    for i in range(n_beams):
        AMCF_codebook_all[i,:] = get_AMCF_beam(AMCF_boundary[i,0], AMCF_boundary[i,1],n_antenna=n_antenna,spacing=spacing)
    return AMCF_codebook_all

def pow_2_dB(x):
    return 10*np.log10(x)
def dB_2_pow(x):
    return 10**(x/10)

class Node():
    def __init__(self, n_antenna:int, n_beam:int, codebook:np.ndarray, beam_index:np.ndarray, noise_power=0):
        super(Node, self).__init__()
        self.codebook = codebook
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.beam_index = beam_index # indices of the beams (in the same layer) in the codebook
        self.noise_power = noise_power
        self.parent = None
        self.child = None
        
    def forward(self, h):
        bf_signal = np.matmul(h, self.codebook.conj().T)
        noise_real = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        noise_imag = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        bf_signal = bf_signal + noise_real + 1j*noise_imag
        bf_gain = np.power(np.absolute(bf_signal),2)
        # bf_gain = np.power(np.absolute(np.matmul(h, self.codebook.conj().T)),2)
        return bf_gain
    
    def get_beam_index(self):
        return self.beam_index

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return self.get_child() is None
    
    def is_root(self):
        return self.get_parent() is None
    
class Beam_Search_Tree():
    def __init__(self, n_antenna, n_narrow_beam, k, noise_power):
        super(Beam_Search_Tree, self).__init__()
        assert math.log(n_narrow_beam,k).is_integer()
        self.n_antenna = n_antenna
        self.k = k #number of beams per branch per layer
        self.n_layer = int(math.log(n_narrow_beam,k))
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = noise_power
        self.beam_search_candidates = []
        for l in range(self.n_layer):
            self.beam_search_candidates.append([])
        self.nodes = []
        for l in range(self.n_layer):
            n_nodes = k**l
            n_beams = k**(l+1)
            if l == self.n_layer-1:
                beams = ULA_DFT_codebook(nseg=n_beams,n_antenna=n_antenna)
            else:                    
                beam_boundaries = AMCF_boundaries(n_beams)
                beams = np.array([get_AMCF_beam(omega_min=beam_boundaries[i,0], omega_max=beam_boundaries[i,1], n_antenna = n_antenna) for i in range(n_beams)])
                beams = np.flipud(beams)
            beam_idx_per_codebook = [np.arange(i,i+k) for i in np.arange(0,n_beams,k)]
            codebooks = [beams[beam_idx_per_codebook[i]] for i in range(n_nodes)]
            nodes_cur_layer = []
            nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k, codebook=codebooks[i], beam_index=beam_idx_per_codebook[i], noise_power=self.noise_power) for i in range(n_nodes)]
            self.nodes.append(nodes_cur_layer)
            if l > 0:
                parent_nodes = self.nodes[l-1]
                for p_i, p_n in enumerate(parent_nodes):
                    child_nodes = nodes_cur_layer[p_i*k:(p_i+1)*k]
                    p_n.set_child(child_nodes)
                    for c_n in child_nodes:
                        c_n.set_parent(p_n)
        self.root = self.nodes[0][0]
        
    def forward(self, h):
        cur_node = self.root
        while not cur_node.is_leaf():
            bf_gain = cur_node.forward(h)
            next_node_idx = bf_gain.argmax()
            cur_node = cur_node.get_child()[next_node_idx]
        nb_bf_gain = cur_node.forward(h)
        max_nb_bf_gain = nb_bf_gain.max()
        max_nb_idx_local = nb_bf_gain.argmax()
        max_nb_idx_global = cur_node.get_beam_index()[max_nb_idx_local]
        return max_nb_bf_gain, max_nb_idx_global        
        
    def forward_batch(self, hbatch):
        bsize, in_dim = hbatch.shape
        max_nb_idx_batch = np.zeros(bsize,dtype=np.int32)
        max_nb_bf_gain_batch = np.zeros(bsize)
        for b_idx in range(bsize):
            h = hbatch[b_idx]
            nb_gain,nb_idx = self.forward(h)
            max_nb_idx_batch[b_idx] = nb_idx
            max_nb_bf_gain_batch[b_idx] = nb_gain
        return max_nb_bf_gain_batch, max_nb_idx_batch
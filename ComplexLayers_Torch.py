# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:49:09 2021

@author: Yuqiang (Ethan) Heng
"""
import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn
from beam_utils import codebook_blockmatrix

    
class PhaseShifter(Module):
    """
    This module is a pytorch implementation of the complex Dense layer for phase shifters satisfying constant modulus constraint
    It is insipred by the implemention by the orignal authors of the Deep Complex Networks paper
    https://github.com/ChihebTrabelsi/deep_complex_networks
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: scaling factor to account for tx and noise power
        theta: the initial phase shift values
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float
    theta: Tensor

    def __init__(self, in_features: int, out_features: int, scale: float=1, theta = None) -> None:
        super(PhaseShifter, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.reset_parameters(theta)

    def reset_parameters(self, theta = None) -> None:
        if theta is None:
            init.uniform_(self.theta, a=0, b=2*np.pi)
        else:
            assert theta.shape == (self.in_dim,self.out_features)
            self.theta = Parameter(theta) 
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        output = torch.matmul(inputs, cat_kernels_4_complex)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def get_theta(self) -> torch.Tensor:
        return self.theta.detach().clone()
    
    def get_weights(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
            imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights

class ComputePower(Module):
    def __init__(self, in_shape):
        super(ComputePower, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        return abs_values
    
class Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size())*torch.sqrt(torch.tensor([self.noise_power/2]))/torch.tensor([self.norm_factor])
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            return self.complex_codebook

def conv_dimension(input_dim,kernel_size,padding=0,dilation=1,stride=1):
    return np.floor((input_dim+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    
class CNN_Classifier(nn.Module):
    def __init__(self, n_wide_beam, n_narrow_beam):
        super(CNN_Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        output_dim = conv_dimension(input_dim=n_wide_beam*2,kernel_size=3)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=3)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=3)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=2)
        self.flatten_dim = 64*int(output_dim)
        self.dense1 = nn.Linear(in_features=self.flatten_dim,out_features=n_narrow_beam)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dense1(x)
        return x
        
class Beam_Classifier_CNN(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier_CNN, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.CNN_Classifier = CNN_Classifier(n_wide_beam, n_narrow_beam)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size())*torch.sqrt(torch.tensor([self.noise_power/2]))/torch.tensor([self.norm_factor])
        bf_signal = bf_signal + noise_vec
        bf_signal = bf_signal.unsqueeze(1)
        out = self.CNN_Classifier(bf_signal)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            return self.complex_codebook  

class RSRP_CNN_Classifier(nn.Module):
    def __init__(self, n_wide_beam, n_narrow_beam):
        super(RSRP_CNN_Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=32,kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        output_dim = conv_dimension(input_dim=n_wide_beam,kernel_size=2)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=1)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=2)
        output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=1)
        self.flatten_dim = 64*int(output_dim)
        self.dense1 = nn.Linear(in_features=self.flatten_dim,out_features=n_narrow_beam)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))        
        x = self.pool(self.relu(self.conv2(x)))        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dense1(x)
        return x

class Beam_Classifier_CNN_RSRP(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier_CNN_RSRP, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.CNN_Classifier = RSRP_CNN_Classifier(n_wide_beam, n_narrow_beam)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size())*torch.sqrt(torch.tensor([self.noise_power/2]))/torch.tensor([self.norm_factor])
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        bf_power = bf_power.unsqueeze(1)
        out = self.CNN_Classifier(bf_power)
        return out
    
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            return self.complex_codebook 
        
def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
            val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,train_loss,train_acc,val_loss,val_acc))
    return train_loss_hist, val_loss_hist

def eval_model(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    test_acc = 0
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        var_X_batch = X_batch.float()
        var_y_batch = y_batch.long ()  
        output = model(var_X_batch)
        loss = loss_fn(output, var_y_batch)
        test_loss += loss.detach().item()
        test_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
    test_loss /= batch_idx + 1
    test_acc /= batch_idx + 1
    return test_loss, test_acc
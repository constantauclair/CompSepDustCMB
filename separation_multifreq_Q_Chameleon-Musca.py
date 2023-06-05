# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import scipy.stats as stats

'''
This component separation algorithm aims to separate the Stokes Q parameter of the polarized dust emission from the CMB 
and noise contamination on Planck-like mock data. This is done at 217 and 353 GHz. 
It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

Loss terms:
# Auto statistics
L1 : u_dust + CMB + n = d
L2 : u_CMB = CMB
L3 : d - u_dust - u_CMB = n

# Cross-frequency statistics
L4 : (u_dust_1 + CMB + n_1)*(u_dust_2 + CMB + n_2) = d_1 * d_2

# Cross-component statistics
L5 : u_dust * u_CMB = 0
L6 : u_dust * n = 0
L7 : u_CMB * n = 0

# Correlation with T
L8 : u_CMB * T_CMB = CMB * T_CMB
'''
#######
# INPUT PARAMETERS
#######

n_freq = 2
n_maps = n_freq+1

M, N = 512, 512
J = 7
L = 4
dn = 5
pbc = True
method = 'L-BFGS-B'
losses = ['L1','L4']

slope = -4

file_name="separation_multifreq_Q_Chameleon-Musca_L14_fbm-4.npy"

n_step1 = 5
iter_per_step1 = 50
    
n_step2 = 5#10
iter_per_step2 = 50#100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 100 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

#######
# DATA
#######

Dust_1 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Dust_IQU_217.npy')[1]#[::2,::2]
Dust_2 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Dust_IQU_353.npy')[1]#[::2,::2]
    
CMB = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[1,0]#[::2,::2]
    
CMB_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[1]#[:,::2,::2]

Noise_1 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_217.npy')[1,0]#[::2,::2]
Noise_2 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_353.npy')[1,0]#[::2,::2]
    
Noise_1_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_217.npy')[1]#[:,::2,::2]
Noise_2_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_353.npy')[1]#[:,::2,::2]

TCMB = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[0,0]#[::2,::2]

TCMB_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[0]#[:,::2,::2]

Mixture_1 = Dust_1 + CMB + Noise_1
Mixture_2 = Dust_2 + CMB + Noise_2

print("SNR F1 =",np.std(Dust_1)/np.std(CMB+Noise_1))
print("SNR F2 =",np.std(Dust_2)/np.std(CMB+Noise_2))

## Define final variables

Mixture = np.array([Mixture_1,Mixture_2])

Dust = np.array([Dust_1,Dust_2])

Noise = np.array([Noise_1,Noise_2])

Noise_syn = np.array([Noise_1_syn,Noise_2_syn])

#######
# USEFUL FUNCTIONS
#######

def power_spectrum(image):
    assert image.shape[0] == image.shape[1]    
    n = image.shape[0]
    fourier = np.fft.fftn(image)
    amplitude = (np.abs(fourier) ** 2).flatten()
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten() ** (1 / 2)
    kbins = np.arange(1 / 2, n // 2 + 1, 1)
    kvals = (kbins[1:] + kbins[:-1]) / 2
    bins, _, _ = stats.binned_statistic(knrm, amplitude, statistic = "mean", bins = kbins)
    return kvals, bins

def generate_partial_fbm(noisy_data,noise,slope,lim=3):
    noise_k, noise_bins = power_spectrum(noise)
    noisy_data_k, noisy_data_bins = power_spectrum(noisy_data)
    clean_mask = noise_bins < noisy_data_bins/lim
    k_clean = noisy_data_k[clean_mask]
    bins_clean = noisy_data_bins[clean_mask]
    n = np.shape(noisy_data)[0]
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2) ** (1 / 2)
    filtered_k_mask = knrm > k_clean[-1]
    noisy_data_FT = np.fft.fft2(noisy_data)
    random_phases = np.exp(1j*np.angle(np.fft.fft2(np.random.random(size=np.shape(noisy_data)))))
    if type(slope) == list:
        several_new_data = np.zeros((len(slope),n,n))
        for i in range(len(slope)):
            new_data_FT = np.where(filtered_k_mask,np.sqrt(bins_clean[-1])*random_phases*(knrm/k_clean[-1])**(slope[i]/2),noisy_data_FT)
            several_new_data[i] = np.real(np.fft.ifft2(new_data_FT))
        return several_new_data
    else:
        new_data_FT = np.where(filtered_k_mask,np.sqrt(bins_clean[-1])*random_phases*(knrm/k_clean[-1])**(slope/2),noisy_data_FT)
        new_data = np.real(np.fft.ifft2(new_data_FT))
        return new_data
    
def create_batch(n_freq, n, device):
    batch = torch.zeros([n_freq,n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[:,i] = n[:,i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

def create_mono_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

Noise_batch = create_batch(n_freq, torch.from_numpy(Noise_syn).to(device), device=device)
CMB_batch = create_mono_batch(torch.from_numpy(CMB_syn).to(device), device=device)
TCMB_batch = create_mono_batch(torch.from_numpy(TCMB_syn).to(device), device=device)

def compute_coeffs_mean_std(mode,contamination_batch,cross_contamination_batch=None,x=None,real_imag=True):
    coeffs_number = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).size(-1)
    ref_type = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).type()
    # Mode for L1 
    if mode == 'classic_bias':
        COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=ref_type)
        coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((n_freq,batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(x.unsqueeze(1).expand(contamination_batch[:,i].size()) + contamination_batch[:,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[:,indices].unsqueeze(1).expand(coeffs_chunk.size()).type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[:,computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=1)
            std = torch.std(COEFFS,axis=1)
    # Mode for L2
    if mode == 'mean_monofreq':
        COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(contamination_batch[i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=0)
            std = torch.std(COEFFS,axis=0)
    # Mode for L3
    if mode == 'mean':
        COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((n_freq,batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(contamination_batch[:,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,:,indices] = coeffs_chunk.type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[:,computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=1)
            std = torch.std(COEFFS,axis=1)
    # Mode for L4
    if mode == 'cross_freq_bias':
        COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
        coeffs_ref = wph_op.apply([x[0],x[1]], norm=None, pbc=pbc, cross=True).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure([x[0] + contamination_batch[0,i],x[1] + contamination_batch[1,i]], pbc=pbc, cross=True)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
                batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices]
                del coeffs_chunk, indices
            COEFFS[computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=0)
            std = torch.std(COEFFS,axis=0)
    # Mode for L5, L6, L7
    if mode == 'cross_mean':
        COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((n_freq,batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure([contamination_batch[:,i],cross_contamination_batch[:,i]], pbc=pbc, cross=True)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
                batch_COEFFS[:,:,indices] = coeffs_chunk.type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[:,computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=1)
            std = torch.std(COEFFS,axis=1)
    # Mode for L8
    if mode == 'cross_mean_mono':
        COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((n_freq,batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure([contamination_batch[:,i],cross_contamination_batch[:,i]], pbc=pbc, cross=True)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
                batch_COEFFS[:,:,indices] = coeffs_chunk.type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[:,computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=0)
            std = torch.std(COEFFS,axis=0)
    return bias.to(device), std.to(device)

def compute_mask(x,std,real_imag=True,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    if not real_imag:
        mask = torch.logical_and(torch.abs(coeffs).to(device) > 1e-7, torch.abs(std).to(device) > 0)
    if real_imag:
        mask_real = torch.logical_and(torch.real(coeffs).to(device) > 1e-7, std[0].to(device) > 0)
        mask_imag = torch.logical_and(torch.imag(coeffs).to(device) > 1e-7, std[1].to(device) > 0)
        mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_loss(mode,x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    # Mode for the first iteration step
    if mode == 'first_iter':
        loss_tot_F1 = torch.zeros(1)
        loss_tot_F2 = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
            # Loss F1
            loss_F1 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[0])[mask[0,0,indices]] - coeffs_target[0,0][indices][mask[0,0,indices]]) / std[0,0][indices][mask[0,0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[0])[mask[1,0,indices]] - coeffs_target[1,0][indices][mask[1,0,indices]]) / std[1,0][indices][mask[1,0,indices]] ) ** 2) ) / ( mask[0,0].sum() + mask[1,0].sum() )
            loss_F1.backward(retain_graph=True)
            loss_tot_F1 += loss_F1.detach().cpu()
            # Loss F2
            loss_F2 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[1])[mask[0,1,indices]] - coeffs_target[0,1][indices][mask[0,1,indices]]) / std[0,1][indices][mask[0,1,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[1])[mask[1,1,indices]] - coeffs_target[1,1][indices][mask[1,1,indices]]) / std[1,1][indices][mask[1,1,indices]] ) ** 2) ) / ( mask[0,1].sum() + mask[1,1].sum() )
            loss_F2.backward(retain_graph=True)
            loss_tot_F2 += loss_F2.detach().cpu()
            #
            del coeffs_chunk, indices, loss_F1, loss_F2
        return loss_tot_F1, loss_tot_F2
    # Mode for L1 and L3
    if mode in ['L1','L3']:
        loss_tot_F1 = torch.zeros(1)
        loss_tot_F2 = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
            # Loss F1
            loss_F1 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[0])[mask[0,0,indices]] - coeffs_target[0,0][indices][mask[0,0,indices]]) / std[0,0][indices][mask[0,0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[0])[mask[1,0,indices]] - coeffs_target[1,0][indices][mask[1,0,indices]]) / std[1,0][indices][mask[1,0,indices]] ) ** 2) ) / ( mask[0,0].sum() + mask[1,0].sum() )
            loss_F1.backward(retain_graph=True)
            loss_tot_F1 += loss_F1.detach().cpu()
            # Loss F2
            loss_F2 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[1])[mask[0,1,indices]] - coeffs_target[0,1][indices][mask[0,1,indices]]) / std[0,1][indices][mask[0,1,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[1])[mask[1,1,indices]] - coeffs_target[1,1][indices][mask[1,1,indices]]) / std[1,1][indices][mask[1,1,indices]] ) ** 2) ) / ( mask[0,1].sum() + mask[1,1].sum() )
            loss_F2.backward(retain_graph=True)
            loss_tot_F2 += loss_F2.detach().cpu()
            #
            del coeffs_chunk, indices, loss_F1, loss_F2
        return loss_tot_F1, loss_tot_F2
    # Mode for L2
    if mode in ['L2']:
        loss_tot = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() )
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        return loss_tot
    # Mode for L4 and L8
    if mode in ['L4','L8']:
        loss_tot = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc, cross=True)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() )
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        return loss_tot
    # Mode for L5, L6 and L7
    if mode in ['L5','L6','L7']:
        loss_tot_F1 = torch.zeros(1)
        loss_tot_F2 = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, cross=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, cross=True, pbc=pbc)
            # Loss F1
            loss_F1 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[0])[mask[0,0,indices]] - coeffs_target[0,0][indices][mask[0,0,indices]]) / std[0,0][indices][mask[0,0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[0])[mask[1,0,indices]] - coeffs_target[1,0][indices][mask[1,0,indices]]) / std[1,0][indices][mask[1,0,indices]] ) ** 2) ) / ( mask[0,0].sum() + mask[1,0].sum() )
            loss_F1.backward(retain_graph=True)
            loss_tot_F1 += loss_F1.detach().cpu()
            # Loss F2
            loss_F2 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[1])[mask[0,1,indices]] - coeffs_target[0,1][indices][mask[0,1,indices]]) / std[0,1][indices][mask[0,1,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[1])[mask[1,1,indices]] - coeffs_target[1,1][indices][mask[1,1,indices]]) / std[1,1][indices][mask[1,1,indices]] ) ** 2) ) / ( mask[0,1].sum() + mask[1,1].sum() )
            loss_F2.backward(retain_graph=True)
            loss_tot_F2 += loss_F2.detach().cpu()
            #
            del coeffs_chunk, indices, loss_F1, loss_F2
        return loss_tot_F1, loss_tot_F2

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_freq, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss
    loss_tot_F1, loss_tot_F2 = compute_loss('first_iter',u,coeffs_target,std,mask)
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_F1 + loss_tot_F2
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L F1 = "+str(round(loss_tot_F1.item(),3)))
    print("L F2 = "+str(round(loss_tot_F2.item(),3)))
    print("")
    
    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_maps, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    u_dust = u[:n_freq]
    u_CMB = u[n_freq]
    
    # Define total loss 
    L = 0
    
    # Compute the losses
    if 'L1' in losses:
        L1_F1, L1_F2 = compute_loss('L1',u_dust,coeffs_target_L1,std_L1,mask_L1)
        L = L + L1_F1 + L1_F2
    if 'L2' in losses:
        L2 = compute_loss('L2',u_CMB,coeffs_target_L2,std_L2,mask_L2)
        L = L + L2
    if 'L3' in losses:
        L3_F1, L3_F2 = compute_loss('L3',torch.from_numpy(Mixture).to(device) - u_dust - u_CMB,coeffs_target_L3,std_L3,mask_L3)
        L = L + L3_F1 + L3_F2
    if 'L4' in losses:
        L4 = compute_loss('L4',[u_dust[0],u_dust[1]],coeffs_target_L4,std_L4,mask_L4)
        L = L + L4
    if 'L5' in losses:
        L5_F1, L5_F2 = compute_loss('L5',[u_dust,u_CMB.expand((n_freq,M,N))],coeffs_target_L5,std_L5,mask_L5)
        L = L + L5_F1 + L5_F2
    if 'L6' in losses:
        L6_F1, L6_F2 = compute_loss('L6',[u_dust,torch.from_numpy(Mixture).to(device) - u_dust - u_CMB],coeffs_target_L6,std_L6,mask_L6)
        L = L + L6_F1 + L6_F2
    if 'L7' in losses:
        L7_F1, L7_F2 = compute_loss('L7',[u_CMB.expand((n_freq,M,N)),torch.from_numpy(Mixture).to(device) - u_dust - u_CMB],coeffs_target_L7,std_L7,mask_L7)
        L = L + L7_F1 + L7_F2
    if 'L8' in losses:
        L8 = compute_loss('L8',[u_CMB,torch.from_numpy(TCMB).to(device)],coeffs_target_L8,std_L8,mask_L8)
        L = L + L8
        
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    if 'L1' in losses:
        print("L1 F1 = "+str(round(L1_F1.item(),3)))
        print("L1 F2 = "+str(round(L1_F2.item(),3)))
    if 'L2' in losses:
        print("L2 = "+str(round(L2.item(),3)))
    if 'L3' in losses:
        print("L3 F1 = "+str(round(L3_F1.item(),3)))
        print("L3 F2 = "+str(round(L3_F2.item(),3)))
    if 'L4' in losses:
        print("L4 = "+str(round(L4.item(),3)))
    if 'L5' in losses:
        print("L5 F1 = "+str(round(L5_F1.item(),3)))
        print("L5 F2 = "+str(round(L5_F2.item(),3)))
    if 'L6' in losses:
        print("L6 F1 = "+str(round(L6_F1.item(),3)))
        print("L6 F2 = "+str(round(L6_F2.item(),3)))
    if 'L7' in losses:
        print("L7 F1 = "+str(round(L7_F1.item(),3)))
        print("L7 F2 = "+str(round(L7_F2.item(),3)))
    if 'L8' in losses:
        print("L8 = "+str(round(L8.item(),3)))
    print("")

    eval_cnt += 1
    return L.item(), u_grad.ravel()

#######
# MINIMIZATION
#######

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization (only S11)...")
    
    eval_cnt = 0
    
    Initial_condition = np.array([generate_partial_fbm(Mixture_1,Noise_1+CMB,slope),generate_partial_fbm(Mixture_2,Noise_2+CMB,slope)])
    
    Dust_tilde0 = Initial_condition
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias, std = compute_coeffs_mean_std('classic_bias',Noise_batch+CMB_batch.expand(Noise_batch.size()),x=Dust_tilde0)
        
        # Mask coputation
        mask = compute_mask(Dust_tilde0, std).to(device)
        
        # Coeffs target computation
        coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias[1],dim=0)))
        
        # Minimization
        result = opt.minimize(objective1, torch.from_numpy(Initial_condition).ravel(), method=method, jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((n_freq, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    
    # Creating new set of variables
    Current_maps0 = np.array([Dust_tilde0[0],Dust_tilde0[1],np.random.normal(np.mean(CMB),np.std(CMB),size=(M,N))])
    
    # Computation of the coeffs and std
    if 'L2' in losses:
        coeffs_target_L2, std_L2 = compute_coeffs_mean_std('mean_monofreq', CMB_batch)
    if 'L3' in losses:
        coeffs_target_L3, std_L3 = compute_coeffs_mean_std('mean', Noise_batch)
    if 'L7' in losses:
        coeffs_target_L7, std_L7 = compute_coeffs_mean_std('cross_mean', CMB_batch.expand(Noise_batch.size()), cross_contamination_batch=Noise_batch)
    if 'L8' in losses:
        coeffs_target_L8, std_L8 = compute_coeffs_mean_std('cross_mean_mono', CMB_batch, cross_contamination_batch=TCMB_batch)
    
    # Mask computation
    if 'L2' in losses:
        mask_L2 = compute_mask(torch.from_numpy(CMB).to(device),std_L2)
    if 'L3' in losses:
        mask_L3 = compute_mask(torch.from_numpy(Noise).to(device),std_L3)
    if 'L7' in losses:
        mask_L7 = compute_mask([torch.from_numpy(CMB).to(device).expand((n_freq,M,N)),torch.from_numpy(Noise).to(device)],std_L7,cross=True)
    if 'L8' in losses:
        mask_L8 = compute_mask([torch.from_numpy(CMB).to(device),torch.from_numpy(TCMB).to(device)],std_L8,cross=True)
    
    Current_maps = Current_maps0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Current_maps = torch.from_numpy(Current_maps).to(device)
        
        # Bias computation
        if 'L1' in losses:
            bias_L1, std_L1 = compute_coeffs_mean_std('classic_bias', Noise_batch+CMB_batch.expand(Noise_batch.size()), x=Current_maps[:n_freq])
        if 'L4' in losses:
            bias_L4, std_L4 = compute_coeffs_mean_std('cross_freq_bias', Noise_batch, x=Current_maps[:n_freq])
        if 'L5' in losses:
            coeffs_target_L5, std_L5 = compute_coeffs_mean_std('cross_mean', Current_maps[:n_freq].expand((n_freq,n_batch,batch_size,M,N)), cross_contamination_batch=CMB_batch.expand((n_freq,n_batch,batch_size,M,N)))
        if 'L6' in losses:
            coeffs_target_L6, std_L6 = compute_coeffs_mean_std('cross_mean', Current_maps[:n_freq].expand((n_freq,n_batch,batch_size,M,N)), cross_contamination_batch=Noise_batch)
        
        # Coeffs target computation
        if 'L1' in losses:
            coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias_L1[1],dim=0)))
        if 'L4' in losses:
            coeffs_dd = wph_op.apply([torch.from_numpy(Mixture[0]),torch.from_numpy(Mixture[1])], norm=None, cross=True, pbc=pbc)
            coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_dd) - bias_L4[0],dim=0),torch.unsqueeze(torch.imag(coeffs_dd) - bias_L4[1],dim=0)))
        
        # Mask computation
        if 'L1' in losses:
            mask_L1 = compute_mask(Current_maps[:n_freq],std_L1)
        if 'L4' in losses:
            mask_L4 = compute_mask([Current_maps[0],Current_maps[1]],std_L4,cross=True)
        if 'L5' in losses:
            mask_L5 = compute_mask([Current_maps[:n_freq],Current_maps[2].expand((2,M,N))],std_L5,cross=True)
        if 'L6' in losses:
            mask_L6 = compute_mask([Current_maps[:n_freq],torch.from_numpy(Noise).to(device)],std_L6,cross=True)
        
        # Minimization
        result = opt.minimize(objective2, torch.from_numpy(np.array([Initial_condition[0],Initial_condition[1],Current_maps0[2]])).ravel(), method=method, jac=True, tol=None, options=optim_params2)
        final_loss, Current_maps, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Current_maps = Current_maps.reshape((n_maps, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,np.array([CMB,CMB]),np.array([TCMB,TCMB]),Noise,Current_maps[:n_freq],np.array([Current_maps[n_freq],Current_maps[n_freq]]),Mixture-Current_maps[:n_freq]-np.array([Current_maps[n_freq],Current_maps[n_freq]]),Current_maps0[:n_freq],Initial_condition])        
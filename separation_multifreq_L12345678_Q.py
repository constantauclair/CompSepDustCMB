# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw

#######
# INPUT PARAMETERS
#######

### Loss terms

# Auto statistics
# L1 : u_dust + CMB + n = d
# L2 : u_CMB = CMB
# L3 : d - u_dust - u_CMB = n

# Cross-frequency statistics
# L4 : (u_dust_1 + CMB + n_1)*(u_dust_2 + CMB + n_2) = d_1 * d_2

# Cross-component statistics
# L5 : u_dust * u_CMB = 0
# L6 : u_dust * n = 0
# L7 : u_CMB * n = 0

# Correlation with T
# L8 : u_CMB * T_CMB = CMB * T_CMB
###

###
n_freq = 2
n_maps = n_freq+1

M, N = 512,512
J = 7
L = 4
dn = 0
pbc = True

file_name="separation_multifreq_L12345678_Q.npy"

n_step1 = 5
iter_per_step1 = 50
    
n_step2 = 10
iter_per_step2 = 100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 100 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

#######
# DATA
#######

# Dust_1 = np.load('data/IQU_Planck_data/Dust_Q_217.npy')
# Dust_2 = np.load('data/IQU_Planck_data/Dust_Q_353.npy')
    
# CMB_1 = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1,0]
# CMB_2 = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1,0]
    
# CMB_1_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1]
# CMB_2_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1]

# Noise_1 = np.load('data/IQU_Planck_data/Noise_QU_217.npy')[0,0]
# Noise_2 = np.load('data/IQU_Planck_data/Noise_QU_353.npy')[0,0]
    
# Noise_1_syn = np.load('data/IQU_Planck_data/Noise_QU_217.npy')[0]
# Noise_2_syn = np.load('data/IQU_Planck_data/Noise_QU_353.npy')[0]

# TCMB = np.load('data/IQU_Planck_data/CMB_IQU.npy')[0,0]

# TCMB_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[0]

#####################

Dust_1 = np.load('data/IQU_Planck_data/Dust_IQU_217.npy')[1]
Dust_2 = np.load('data/IQU_Planck_data/Dust_IQU_353.npy')[1] 
    
CMB_1 = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1,0]
CMB_2 = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1,0]
    
CMB_1_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1]
CMB_2_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[1]

Noise_1 = np.load('data/IQU_Planck_data/Noise_IQU_217.npy')[1,0]
Noise_2 = np.load('data/IQU_Planck_data/Noise_IQU_353.npy')[1,0]
    
Noise_1_syn = np.load('data/IQU_Planck_data/Noise_IQU_217.npy')[1]
Noise_2_syn = np.load('data/IQU_Planck_data/Noise_IQU_353.npy')[1]

TCMB = np.load('data/IQU_Planck_data/CMB_IQU.npy')[0,0]

TCMB_syn = np.load('data/IQU_Planck_data/CMB_IQU.npy')[0]

#####################

Mixture_1 = Dust_1 + CMB_1 + Noise_1
Mixture_2 = Dust_2 + CMB_2 + Noise_2

print("SNR F1 =",np.std(Dust_1)/np.std(CMB_1+Noise_1))
print("SNR F2 =",np.std(Dust_2)/np.std(CMB_2+Noise_2))

## Define final variables

Mixture = np.array([Mixture_1,Mixture_2])

Dust = np.array([Dust_1,Dust_2])

CMB = np.array([CMB_1,CMB_2])

Noise = np.array([Noise_1,Noise_2])

CMB_syn = np.array([CMB_1_syn,CMB_2_syn])

Noise_syn = np.array([Noise_1_syn,Noise_2_syn])

#######
# USEFUL FUNCTIONS
#######

def create_batch(n_freq, n_maps, n, device, batch_size):
    x = n_maps//batch_size
    if n_maps % batch_size != 0:
        batch = torch.zeros([n_freq,x+1,batch_size,M,N])
        for i in range(x):
            batch[:,i] = n[:,i*batch_size:(i+1)*batch_size,:,:]
        batch[:,x] = n[:,x*batch_size:,:,:]
    else:
        batch = torch.zeros([n_freq,x,batch_size,M,N])
        for i in range(x):
            batch[:,i] = n[:,i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

def create_mono_batch(n_maps, n, device, batch_size):
    x = n_maps//batch_size
    if n_maps % batch_size != 0:
        batch = torch.zeros([x+1,batch_size,M,N])
        for i in range(x):
            batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
        batch[x] = n[x*batch_size:,:,:]
    else:
        batch = torch.zeros([x,batch_size,M,N])
        for i in range(x):
            batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

Noise_batch = create_batch(n_freq, Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
CMB_batch = create_batch(n_freq, Mn, torch.from_numpy(CMB_syn).to(device), device=device, batch_size=batch_size)
TCMB_batch = create_mono_batch(Mn, torch.from_numpy(TCMB_syn).to(device), device=device, batch_size=batch_size)

def compute_bias_std_L1(x):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(Noise_batch.shape[1]):
        this_batch_size = len(Noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + CMB_batch[freq,i] + Noise_batch[freq,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk - coeffs_ref[freq,indices]
                del coeffs_chunk, indices
            COEFFS[freq,computed_noise:computed_noise+this_batch_size] = batch_COEFFS[freq]
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=1)
    std = torch.std(COEFFS,axis=1)
    return bias, std

def compute_complex_bias_std_L1(x):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(Noise_batch.shape[1]):
        this_batch_size = len(Noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + CMB_batch[freq,i] + Noise_batch[freq,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk - coeffs_ref[freq,indices]
                del coeffs_chunk, indices
            COEFFS[freq,computed_noise:computed_noise+this_batch_size] = batch_COEFFS[freq]
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return bias, std

def compute_complex_mean_std_L2():
    coeffs_ref = wph_op.apply(torch.from_numpy(CMB_1).to(device), norm=None, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_CMB = 0
    for i in range(CMB_batch.shape[1]):
        this_batch_size = len(CMB_batch[0,i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        cmb, nb_chunks = wph_op.preconfigure(CMB_batch[0,i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(cmb, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
            del coeffs_chunk, indices
        COEFFS[computed_CMB:computed_CMB+this_batch_size] = batch_COEFFS
        computed_CMB += this_batch_size
        del cmb, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return mean, std

def compute_complex_mean_std_L3():
    coeffs_ref = wph_op.apply(Noise, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(Noise_batch.shape[1]):
        this_batch_size = len(Noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            noise, nb_chunks = wph_op.preconfigure(Noise_batch[freq,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(noise, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            COEFFS[freq,computed_noise:computed_noise+this_batch_size] = batch_COEFFS[freq]
        computed_noise += this_batch_size
        del noise, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return mean, std

def compute_complex_bias_std_L4(x):
    coeffs_ref = wph_op.apply([x[0],x[1]], norm=None, cross=True, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    n_pairs = int(Noise_batch.shape[1]*(Noise_batch.shape[1]-1)/2 * Noise_batch.shape[2])
    COEFFS = torch.zeros((n_pairs,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_pairs = 0
    for i in range(Noise_batch.shape[1]):
        for j in range(Noise_batch.shape[1]):
            if j>i:
                batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
                uu_noisy, nb_chunks = wph_op.preconfigure([x[0]+CMB_batch[0,i]+Noise_batch[0,i],x[1]+CMB_batch[1,i]+Noise_batch[1,j]], cross=True, pbc=pbc)
                for j in range(nb_chunks):
                    coeffs_chunk, indices = wph_op.apply(uu_noisy, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                    batch_COEFFS[:,indices] = coeffs_chunk - coeffs_ref[indices]
                    del coeffs_chunk, indices
                COEFFS[computed_pairs:computed_pairs+batch_size] = batch_COEFFS
                computed_pairs += batch_size
                del uu_noisy, nb_chunks, batch_COEFFS
                sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias, std

def compute_complex_mean_std_L5(x):
    coeffs_ref = wph_op.apply([x,CMB_syn[:,0]], norm=None, cross=True, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_CMB = 0
    for i in range(CMB_batch.shape[1]):
        this_batch_size = len(CMB_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            dustxCMB, nb_chunks = wph_op.preconfigure([x[freq].expand((this_batch_size,M,N)),CMB_batch[freq,i]], cross=True, pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(dustxCMB, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            COEFFS[freq,computed_CMB:computed_CMB+this_batch_size] = batch_COEFFS[freq]
        computed_CMB += this_batch_size
        del dustxCMB, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return mean, std

def compute_complex_mean_std_L6(x):
    coeffs_ref = wph_op.apply([x,Noise_syn[:,0]], norm=None, cross=True, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(Noise_batch.shape[1]):
        this_batch_size = len(Noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            dustxnoise, nb_chunks = wph_op.preconfigure([x[freq].expand((this_batch_size,M,N)),Noise_batch[freq,i]], cross=True, pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(dustxnoise, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            COEFFS[freq,computed_noise:computed_noise+this_batch_size] = batch_COEFFS[freq]
        computed_noise += this_batch_size
        del dustxnoise, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return mean, std

def compute_complex_mean_std_L7():
    coeffs_ref = wph_op.apply([CMB_syn[:,0],Noise_syn[:,0]], norm=None, cross=True, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(Noise_batch.shape[1]):
        this_batch_size = len(Noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            CMBxnoise, nb_chunks = wph_op.preconfigure([CMB_batch[freq,i],Noise_batch[freq,i]], cross=True, pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(CMBxnoise, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                batch_COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            COEFFS[freq,computed_noise:computed_noise+this_batch_size] = batch_COEFFS[freq]
        computed_noise += this_batch_size
        del CMBxnoise, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return mean, std

def compute_complex_mean_std_L8():
    coeffs_ref = wph_op.apply([torch.from_numpy(CMB_1).to(device),torch.from_numpy(TCMB).to(device)], cross=True, norm=None, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_CMB = 0
    for i in range(CMB_batch.shape[1]):
        this_batch_size = len(CMB_batch[0,i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        cmb, nb_chunks = wph_op.preconfigure([CMB_batch[0,i],TCMB_batch[i]], cross=True, pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(cmb, j, cross=True, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
            del coeffs_chunk, indices
        COEFFS[computed_CMB:computed_CMB+this_batch_size] = batch_COEFFS
        computed_CMB += this_batch_size
        del cmb, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return mean, std

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_freq, M, N))
    
    print("F1 =",u[0].mean(),u[0].std())
    print("F2 =",u[1].mean(),u[1].std())
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss
    loss_tot_F1 = torch.zeros(1)
    loss_tot_F2 = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
        loss_F1 = torch.sum(torch.abs( (coeffs_chunk[0] - coeffs_target[0,indices]) / std[0,indices] ) ** 2)
        loss_F2 = torch.sum(torch.abs( (coeffs_chunk[1] - coeffs_target[1,indices]) / std[1,indices] ) ** 2)
        loss_F1 = loss_F1 / len(coeffs_target[0])
        loss_F2 = loss_F2 / len(coeffs_target[1])
        loss_F1.backward(retain_graph=True)
        loss_F2.backward(retain_graph=True)
        loss_tot_F1 += loss_F1.detach().cpu()
        loss_tot_F2 += loss_F2.detach().cpu()
        del coeffs_chunk, indices, loss_F1, loss_F2
    
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
    
    # Compute the loss 1
    loss_tot_1_F1_real = torch.zeros(1)
    loss_tot_1_F2_real = torch.zeros(1)
    loss_tot_1_F1_imag = torch.zeros(1)
    loss_tot_1_F2_imag = torch.zeros(1)
    u_L1, nb_chunks = wph_op.preconfigure(u_dust, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L1, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L1[0,0][indices]) / std_L1[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std_L1[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L1[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / len(indices) #real_coeffs_number_L1
        loss_F1_imag = loss_F1_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L1
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L1[0,1][indices]) / std_L1[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std_L1[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L1[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / len(indices) #real_coeffs_number_L1
        loss_F2_imag = loss_F2_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L1
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_1_F1_real += loss_F1_real.detach().cpu()
        loss_tot_1_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_1_F2_real += loss_F2_real.detach().cpu()
        loss_tot_1_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
    
    # Compute the loss 2
    loss_tot_2_real = torch.zeros(1)
    loss_tot_2_imag = torch.zeros(1)
    u_L2, nb_chunks = wph_op.preconfigure(u_CMB, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L2, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - coeffs_target_L2[0][indices]) / std_L2[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_L2[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - coeffs_target_L2[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / real_coeffs_number_L2
        loss_imag = loss_imag / imag_coeffs_number_L2
        #
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        #
        loss_tot_2_real += loss_real.detach().cpu()
        loss_tot_2_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
        
    # Compute the loss 3
    loss_tot_3_F1_real = torch.zeros(1)
    loss_tot_3_F2_real = torch.zeros(1)
    loss_tot_3_F1_imag = torch.zeros(1)
    loss_tot_3_F2_imag = torch.zeros(1)
    u_L3, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - u_dust - u_CMB, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L3, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L3[0,0][indices]) / std_L3[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L3[indices] / std_L3[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L3[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_L3
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_L3
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L3[0,1][indices]) / std_L3[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L3[indices] / std_L3[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L3[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_L2
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_L2
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_3_F1_real += loss_F1_real.detach().cpu()
        loss_tot_3_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_3_F2_real += loss_F2_real.detach().cpu()
        loss_tot_3_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 4
    loss_tot_4_real = torch.zeros(1)
    loss_tot_4_imag = torch.zeros(1)
    u_L4, nb_chunks = wph_op.preconfigure([u_dust[0],u_dust[1]], cross=True, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L4, i, norm=None, cross=True, ret_indices=True, pbc=pbc)
        #
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - coeffs_target_L4[0][indices]) / std_L4[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L4[indices] / std_L4[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - coeffs_target_L4[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / len(indices) #real_coeffs_number_L3
        loss_imag = loss_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L3
        #
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        #
        loss_tot_4_real += loss_real.detach().cpu()
        loss_tot_4_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
    
    # Compute the loss 5
    loss_tot_5_F1_real = torch.zeros(1)
    loss_tot_5_F2_real = torch.zeros(1)
    loss_tot_5_F1_imag = torch.zeros(1)
    loss_tot_5_F2_imag = torch.zeros(1)
    u_L5, nb_chunks = wph_op.preconfigure([u_dust,u_CMB.expand((n_freq,M,N))], cross=True, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L5, i, norm=None, cross=True, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L5[0,0][indices]) / std_L5[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L5[indices] / std_L5[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L5[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_L5
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_L5
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L5[0,1][indices]) / std_L5[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L5[indices] / std_L5[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L5[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_L5
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_L5
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_5_F1_real += loss_F1_real.detach().cpu()
        loss_tot_5_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_5_F2_real += loss_F2_real.detach().cpu()
        loss_tot_5_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 6
    loss_tot_6_F1_real = torch.zeros(1)
    loss_tot_6_F2_real = torch.zeros(1)
    loss_tot_6_F1_imag = torch.zeros(1)
    loss_tot_6_F2_imag = torch.zeros(1)
    u_L6, nb_chunks = wph_op.preconfigure([u_dust,torch.from_numpy(Mixture).to(device) - u_dust - u_CMB], cross=True, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L6, i, norm=None, cross=True, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L6[0,0][indices]) / std_L6[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L6[indices] / std_L6[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L6[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_L6
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_L6
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L6[0,1][indices]) / std_L6[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L6[indices] / std_L6[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L6[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_L6
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_L6
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_6_F1_real += loss_F1_real.detach().cpu()
        loss_tot_6_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_6_F2_real += loss_F2_real.detach().cpu()
        loss_tot_6_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 7
    loss_tot_7_F1_real = torch.zeros(1)
    loss_tot_7_F2_real = torch.zeros(1)
    loss_tot_7_F1_imag = torch.zeros(1)
    loss_tot_7_F2_imag = torch.zeros(1)
    u_L7, nb_chunks = wph_op.preconfigure([u_CMB.expand((n_freq,M,N)),torch.from_numpy(Mixture).to(device) - u_dust - u_CMB], cross=True, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L7, i, norm=None, cross=True, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L7[0,0][indices]) / std_L7[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L7[indices] / std_L7[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L7[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_L7
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_L7
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L7[0,1][indices]) / std_L7[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L7[indices] / std_L7[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L7[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_L7
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_L7
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_7_F1_real += loss_F1_real.detach().cpu()
        loss_tot_7_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_7_F2_real += loss_F2_real.detach().cpu()
        loss_tot_7_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 8
    loss_tot_8_real = torch.zeros(1)
    loss_tot_8_imag = torch.zeros(1)
    u_L8, nb_chunks = wph_op.preconfigure([u_CMB,torch.from_numpy(TCMB).to(device)], cross=True, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_L8, i, cross=True, norm=None, ret_indices=True, pbc=pbc)
        # Loss
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - coeffs_target_L8[0][indices]) / std_L8[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L8[indices] / std_L8[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - coeffs_target_L8[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / real_coeffs_number_L8
        loss_imag = loss_imag / imag_coeffs_number_L8
        #
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        #
        loss_tot_8_real += loss_real.detach().cpu()
        loss_tot_8_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
        
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_1_F1_real + loss_tot_1_F1_imag + loss_tot_1_F2_real + loss_tot_1_F2_imag + loss_tot_2_real + loss_tot_2_imag + loss_tot_3_F1_real + loss_tot_3_F1_imag + loss_tot_3_F2_real + loss_tot_3_F2_imag + loss_tot_4_real + loss_tot_4_imag + loss_tot_5_F1_real + loss_tot_5_F1_imag + loss_tot_5_F2_real + loss_tot_5_F2_imag + loss_tot_6_F1_real + loss_tot_6_F1_imag + loss_tot_6_F2_real + loss_tot_6_F2_imag + loss_tot_7_F1_real + loss_tot_7_F1_imag + loss_tot_7_F2_real + loss_tot_7_F2_imag + loss_tot_8_real + loss_tot_8_imag
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    # L1
    print("L1 F1 real = "+str(round(loss_tot_1_F1_real.item(),3)))
    print("L1 F1 imag = "+str(round(loss_tot_1_F1_imag.item(),3)))
    print("L1 F2 real = "+str(round(loss_tot_1_F2_real.item(),3)))
    print("L1 F2 imag = "+str(round(loss_tot_1_F2_imag.item(),3)))
    # L2
    print("L2 real = "+str(round(loss_tot_2_real.item(),3)))
    print("L2 imag = "+str(round(loss_tot_2_imag.item(),3)))
    # L3
    print("L3 F1 real = "+str(round(loss_tot_3_F1_real.item(),3)))
    print("L3 F1 imag = "+str(round(loss_tot_3_F1_imag.item(),3)))
    print("L3 F2 real = "+str(round(loss_tot_3_F2_real.item(),3)))
    print("L3 F2 imag = "+str(round(loss_tot_3_F2_imag.item(),3)))
    # L4
    print("L4 real = "+str(round(loss_tot_4_real.item(),3)))
    print("L4 imag = "+str(round(loss_tot_4_imag.item(),3)))
    # L5
    print("L5 F1 real = "+str(round(loss_tot_5_F1_real.item(),3)))
    print("L5 F1 imag = "+str(round(loss_tot_5_F1_imag.item(),3)))
    print("L5 F2 real = "+str(round(loss_tot_5_F2_real.item(),3)))
    print("L5 F2 imag = "+str(round(loss_tot_5_F2_imag.item(),3)))
    # L6
    print("L6 F1 real = "+str(round(loss_tot_6_F1_real.item(),3)))
    print("L6 F1 imag = "+str(round(loss_tot_6_F1_imag.item(),3)))
    print("L6 F2 real = "+str(round(loss_tot_6_F2_real.item(),3)))
    print("L6 F2 imag = "+str(round(loss_tot_6_F2_imag.item(),3)))
    # L7
    print("L7 F1 real = "+str(round(loss_tot_7_F1_real.item(),3)))
    print("L7 F1 imag = "+str(round(loss_tot_7_F1_imag.item(),3)))
    print("L7 F2 real = "+str(round(loss_tot_7_F2_real.item(),3)))
    print("L7 F2 imag = "+str(round(loss_tot_7_F2_imag.item(),3)))
    # L8
    print("L8 real = "+str(round(loss_tot_8_real.item(),3)))
    print("L8 imag = "+str(round(loss_tot_8_imag.item(),3)))
    print("")

    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

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
    
    Dust_tilde0 = np.array([Mixture_1,Mixture_2])
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias, std = compute_bias_std_L1(Dust_tilde0)
        
        # Coeffs target computation
        coeffs_target = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc) - bias # estimation of the unbiased coefficients
        
        # Minimization
        result = opt.minimize(objective1, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((n_freq, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Creating new set of variables
    Current_maps0 = np.array([Dust_tilde0[0],Dust_tilde0[1],np.random.normal(np.mean(CMB_1),np.std(CMB_1),size=(M,N))])
    
    # Identification of the irrelevant imaginary parts of the coeffs
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    coeffs_imag_L1 = torch.imag(wph_op.apply(Current_maps0[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L1 = torch.where(torch.abs(coeffs_imag_L1) > 1e-6,1,0)
    coeffs_imag_L2 = torch.imag(wph_op.apply(CMB[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L2 = torch.where(torch.abs(coeffs_imag_L2) > 1e-6,1,0)
    coeffs_imag_L3 = torch.imag(wph_op.apply(Noise[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L3 = torch.where(torch.abs(coeffs_imag_L3) > 1e-6,1,0)
    coeffs_imag_L4 = torch.imag(wph_op.apply([Current_maps0[0],Current_maps0[1]],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L4 = torch.where(torch.abs(coeffs_imag_L4) > 1e-6,1,0)
    coeffs_imag_L5 = torch.imag(wph_op.apply([Current_maps0[0],Current_maps0[2]],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L5 = torch.where(torch.abs(coeffs_imag_L5) > 1e-6,1,0)
    coeffs_imag_L6 = torch.imag(wph_op.apply([Current_maps0[0],Noise[0]],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L6 = torch.where(torch.abs(coeffs_imag_L6) > 1e-6,1,0)
    coeffs_imag_L7 = torch.imag(wph_op.apply([CMB[0],Noise[0]],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L7 = torch.where(torch.abs(coeffs_imag_L7) > 1e-6,1,0)
    coeffs_imag_L8 = torch.imag(wph_op.apply([CMB[0],TCMB],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L8 = torch.where(torch.abs(coeffs_imag_L8) > 1e-6,1,0)
    
    # Computation of the coeffs and std
    bias_L1, std_L1 = compute_complex_bias_std_L1(torch.from_numpy(Current_maps0[:n_freq]).to(device))
    coeffs_target_L2, std_L2 = compute_complex_mean_std_L2()
    coeffs_target_L3, std_L3 = compute_complex_mean_std_L3()
    bias_L4, std_L4 = compute_complex_bias_std_L4(torch.from_numpy(Current_maps0[:n_freq]).to(device))
    mean_L5, std_L5 = compute_complex_mean_std_L5(torch.from_numpy(Current_maps0[:n_freq]).to(device))
    mean_L6, std_L6 = compute_complex_mean_std_L6(torch.from_numpy(Current_maps0[:n_freq]).to(device))
    coeffs_target_L7, std_L7 = compute_complex_mean_std_L7()
    coeffs_target_L8, std_L8 = compute_complex_mean_std_L8()
    
    # Compute the number of coeffs
    # L1
    real_coeffs_number_L1 = len(torch.real(wph_op.apply(torch.from_numpy(Current_maps0[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_L1 = torch.nan_to_num(relevant_imaginary_coeffs_L1 / std_L1[1,0],nan=0)
    imag_coeffs_number_L1 = torch.where(torch.sum(torch.where(kept_coeffs_L1>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L1>0,1,0))).item()
    # L2
    real_coeffs_number_L2 = len(torch.real(wph_op.apply(torch.from_numpy(CMB[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_L2 = torch.nan_to_num(relevant_imaginary_coeffs_L2 / std_L2[1],nan=0)
    imag_coeffs_number_L2 = torch.where(torch.sum(torch.where(kept_coeffs_L2>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L2>0,1,0))).item()
    # L3
    real_coeffs_number_L3 = len(torch.real(wph_op.apply(torch.from_numpy(Noise[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_L3 = torch.nan_to_num(relevant_imaginary_coeffs_L3 / std_L3[1,0],nan=0)
    imag_coeffs_number_L3 = torch.where(torch.sum(torch.where(kept_coeffs_L3>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L3>0,1,0))).item()
    # L4
    real_coeffs_number_L4 = len(torch.real(wph_op.apply([torch.from_numpy(Current_maps0[0]).to(device),torch.from_numpy(Current_maps0[1]).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L4 = torch.nan_to_num(relevant_imaginary_coeffs_L4 / std_L4[1],nan=0)
    imag_coeffs_number_L4 = torch.where(torch.sum(torch.where(kept_coeffs_L4>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L4>0,1,0))).item()
    # L5
    real_coeffs_number_L5 = len(torch.real(wph_op.apply([torch.from_numpy(Current_maps0[0]).to(device),torch.from_numpy(Current_maps0[2]).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L5 = torch.nan_to_num(relevant_imaginary_coeffs_L5 / std_L5[1,0],nan=0)
    imag_coeffs_number_L5 = torch.where(torch.sum(torch.where(kept_coeffs_L5>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L5>0,1,0))).item()
    # L6
    real_coeffs_number_L6 = len(torch.real(wph_op.apply([torch.from_numpy(Current_maps0[0]).to(device),torch.from_numpy(Noise[0]).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L6 = torch.nan_to_num(relevant_imaginary_coeffs_L6 / std_L6[1,0],nan=0)
    imag_coeffs_number_L6 = torch.where(torch.sum(torch.where(kept_coeffs_L6>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L6>0,1,0))).item()
    # L7
    real_coeffs_number_L7 = len(torch.real(wph_op.apply([torch.from_numpy(CMB[0]).to(device),torch.from_numpy(Noise[0]).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L7 = torch.nan_to_num(relevant_imaginary_coeffs_L7 / std_L7[1,0],nan=0)
    imag_coeffs_number_L7 = torch.where(torch.sum(torch.where(kept_coeffs_L7>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L7>0,1,0))).item()
    # L8
    real_coeffs_number_L8 = len(torch.real(wph_op.apply([torch.from_numpy(CMB[0]).to(device),torch.from_numpy(TCMB).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L8 = torch.nan_to_num(relevant_imaginary_coeffs_L8 / std_L8[1],nan=0)
    imag_coeffs_number_L8 = torch.where(torch.sum(torch.where(kept_coeffs_L8>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L8>0,1,0))).item()
    
    Current_maps = Current_maps0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Current_maps = torch.from_numpy(Current_maps).to(device)
        
        # Bias computation
        bias_L1, std_L1 = compute_complex_bias_std_L1(Current_maps[:n_freq])
        bias_L4, std_L4 = compute_complex_bias_std_L4(Current_maps[:n_freq])
        
        # Coeffs target computation
        # L1
        coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias_L1[1],dim=0))) # estimation of the unbiased coefficients
        # L4
        coeffs_dd = wph_op.apply([torch.from_numpy(Mixture[0]),torch.from_numpy(Mixture[1])], norm=None, cross=True, pbc=pbc)
        coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_dd) - bias_L4[0],dim=0),torch.unsqueeze(torch.imag(coeffs_dd) - bias_L4[1],dim=0))) # estimation of the unbiased coefficients
        # L5
        coeffs_target_L5, std_L5 = compute_complex_mean_std_L5(Current_maps[:n_freq])
        # L6
        coeffs_target_L6, std_L6 = compute_complex_mean_std_L6(Current_maps[:n_freq])
        
        # Minimization
        result = opt.minimize(objective2, torch.from_numpy(Current_maps0).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        final_loss, Current_maps, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Current_maps = Current_maps.reshape((n_maps, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,CMB,np.array([TCMB,TCMB]),Noise,Current_maps[:n_freq],np.array([Current_maps[n_freq],Current_maps[n_freq]]),Mixture-Current_maps[:n_freq]-np.array([Current_maps[n_freq],Current_maps[n_freq]]),Current_maps0[:n_freq]])        
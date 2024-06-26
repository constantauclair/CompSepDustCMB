# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import scattering as scat
import argparse

#######
# INPUT PARAMETERS
#######

parser = argparse.ArgumentParser()
parser.add_argument('index', type=int)
args = parser.parse_args()

label_sep = args.index

M, N = 256, 256
J = 6
L = 4

SNR = 1

file_name="denoising_final_L12_SNR=1_ScatCov_original_"+str(label_sep)+".npy"

n_step1 = 5
iter_per_step1 = 50

n_step2 = 10
iter_per_step2 = 50

print_loss_freq = 1

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = "cuda:0" # GPU to use

Mn = 200 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

## Loading the data

Dust = np.load('../data/I_maps_v2_leo.npy')[0,0][::2,::2]

Noise = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[np.random.randint(Mn)].real

Noise_syn = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[:Mn].real

## Normalizing the data

Dust = (Dust - np.mean(Dust)) / np.std(Dust)

Noise = (Noise - np.mean(Noise)) / np.std(Noise) / SNR

Noise_syn = (Noise_syn - np.mean(Noise_syn)) / np.std(Noise_syn) / SNR

Mixture = Dust + Noise

#######
# USEFUL FUNCTIONS
#######

def create_batch(n_maps, n, device, batch_size):
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

noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)

def compute_bias_std_L1(x, only_S11=False):
    coeffs_ref = st_calc.scattering_cov_constant(x, only_S11=only_S11)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = st_calc.scattering_cov_constant(x + noise_batch[i], only_S11=only_S11) - coeffs_ref
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=0)
    std = torch.std(COEFFS,axis=0)
    return bias, std

def compute_bias_std_L2(only_S11=False):
    coeffs_ref = st_calc.scattering_cov_constant(torch.from_numpy(Noise), only_S11=only_S11)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = st_calc.scattering_cov_constant(noise_batch[i], only_S11=only_S11)
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=0)
    std = torch.std(COEFFS,axis=0)
    return bias, std

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    if eval_cnt % print_loss_freq == 0:
        print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # L1
    loss_tot_1 = torch.sum(torch.abs( (st_calc.scattering_cov_constant(u, only_S11=True) - coeffs_target_L1) / std_L1 ) ** 2) / coeffs_number_S11
    loss_tot_1.backward(retain_graph=True)
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    if eval_cnt % print_loss_freq == 0:
        print("L = "+str(round(loss_tot_1.item(),5))+" (computed in "+str(round(time.time() - start_time,3))+"s)")
        print("")
    
    eval_cnt += 1
    return loss_tot_1.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    if eval_cnt % print_loss_freq == 0:
        print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # L1
    loss_tot_1 = torch.sum(torch.abs( (st_calc.scattering_cov_constant(u) - coeffs_target_L1) / std_L1 ) ** 2) / coeffs_number
    loss_tot_1.backward(retain_graph=True)
    
    # L2
    loss_tot_2 = torch.sum(torch.abs( (st_calc.scattering_cov_constant(torch.from_numpy(Mixture).to(device)-u) - coeffs_target_L2) / std_L2 ) ** 2) / coeffs_number
    loss_tot_2.backward(retain_graph=True)
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_1 + loss_tot_2
    
    if eval_cnt % print_loss_freq == 0:
        print("L = "+str(round(loss_tot.item(),5)))
        print("(computed in "+str(round(time.time() - start_time,3))+"s)")
        print("L1 = "+str(round(loss_tot_1.item(),5)))
        print("L2 = "+str(round(loss_tot_2.item(),5)))
        print("")

    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

#######
# MINIMIZATION
#######

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Building calculator...")
    start_time = time.time()
    st_calc = scat.Scattering2d(M, N, J, L) 
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## Computing coeffs number
    coeffs_number_S11 = len(st_calc.scattering_cov_constant(torch.from_numpy(Mixture),only_S11=True))
    coeffs_number = len(st_calc.scattering_cov_constant(torch.from_numpy(Mixture)))
    
    ## First minimization
    print("Starting first step of minimization (only S11)...")
    
    eval_cnt = 0
    
    Dust_tilde0 = Mixture
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias_L1, std_L1 = compute_bias_std_L1(Dust_tilde0,only_S11=True)
        
        #print("bias L1 =", bias_L1)
        #print("std_L1 =", std_L1)
        
        # Coeffs target computation
        coeffs_target_L1 = st_calc.scattering_cov_constant(torch.from_numpy(Mixture), only_S11=True) - bias_L1 # estimation of the unbiased coefficients
        
        #print("coeffs_target_L1 =", coeffs_target_L1)
        #print("true_coeffs_L1 =", st_calc.scattering_cov_constant(torch.from_numpy(Dust), only_S11=True))
        #print("true-target/true =", (st_calc.scattering_cov_constant(torch.from_numpy(Dust), only_S11=True)-coeffs_target_L1)/st_calc.scattering_cov_constant(torch.from_numpy(Dust), only_S11=True))
        
        # Minimization
        result = opt.minimize(objective1, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((M, N)).astype(np.float32)
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Computation of the coeffs and std
    coeffs_target_L2, std_L2 = compute_bias_std_L2()
    
    Dust_tilde = Dust_tilde0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        # Initialization of the map
        Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
        
        # Bias computation
        bias_L1, std_L1 = compute_bias_std_L1(Dust_tilde)
        
        # Coeffs target computation
        coeffs_d = st_calc.scattering_cov_constant(torch.from_numpy(Mixture))
        coeffs_target_L1 = coeffs_d - bias_L1 # estimation of the unbiased coefficients
        
        # Minimization
        result = opt.minimize(objective2, torch.from_numpy(Dust_tilde0).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde,Dust_tilde0,Mixture-Dust_tilde0])
        
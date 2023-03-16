# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import multiprocessing as mp
from functools import partial
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

""" 
    The denoising algorithm is a simplified version of that of Regaldo-Saint Blancard+2021.
    It supports multi-GPU calculation.
"""

#######
# INPUT PARAMETERS
#######

M, N = 256,256
J = 6
L = 4
dn = 2
pbc = True
histo = False
alpha = 60 # CIB+Noise loss factor

output_filename = "Test_separation_CMB.npy"

Mn = 20

## Loading the data

Dust = np.load('data/realistic_data/Dust_EE_217_microK.npy')

CIB = np.load('data/realistic_data/CMB_EE_8arcmin_microK.npy')[0]

CIB_syn = np.load('data/realistic_data/CMB_EE_8arcmin_microK.npy')[1:Mn+1]

Noise = np.load('data/realistic_data/Noise_EE_217_8arcmin_microK.npy')[0]

Noise_syn = np.load('data/realistic_data/Noise_EE_217_8arcmin_microK.npy')[1:Mn+1]

Mixture = Dust + CIB + Noise

CIB_Noise = CIB+Noise

CIB_Noise_syn = CIB_syn+Noise_syn

norm = None   # Normalization

devices = [0,1] # List of GPUs to use

optim_params0 = {"maxiter": 20, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params = {"maxiter": 100, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

#######
# SEPARATION
#######
    
def compute_std_L1(x):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    coeffs_number = coeffs_ref.size().item()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    u_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(CIB_Noise_syn).to(0), pbc=pbc)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
        COEFFS[:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
        del coeffs_chunk, indices
    del u_noisy, nb_chunks
    std = torch.std(COEFFS,axis=0)
    return std

def objective_per_gpu_first(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_target = coeffs_target.to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the 100 noise maps
    CIBNoisesyn = torch.from_numpy(CIB_Noise_syn[work_list]).to(device)
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure(u + CIBNoisesyn[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs( (coeffs_chunk - coeffs_target[indices]) / std_target[indices] ) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del CIBNoisesyn, u # To free GPU memory
    
    return loss_tot.item(), x_grad

def objective_per_gpu_second(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_tar_1 = coeffs_target[0].to(device_id)
    coeffs_tar_2 = coeffs_target[1].to(device_id)
    std_tar_1 = torch.from_numpy(coeffs_target[2]).to(device_id)
    std_tar_2 = torch.from_numpy(coeffs_target[3]).to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the 100 noise maps
    CIBNoisesyn = torch.from_numpy(CIB_Noise_syn[work_list]).to(device)
    
    # Compute the loss for dust
    loss_tot1 = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure(u + CIBNoisesyn[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs( (coeffs_chunk - coeffs_tar_1[indices]) / std_tar_1[indices] ) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot1 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Compute the loss for CIB+Noise
    loss_tot2 = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - u, pbc=pbc)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs( (coeffs_chunk - coeffs_tar_2[indices]) / std_tar_2[indices] ) ** 2) / Mn
        loss = loss*alpha
        loss.backward(retain_graph=True)
        loss_tot2 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    sys.stdout.flush() # Flush the standard output
    
    if device == 0:
        print("L1 =",loss_tot1)
        print("L2 =",loss_tot2)
        
    # Total loss
    Total_loss = loss_tot1 + loss_tot2
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del CIBNoisesyn, u # To free GPU memory
    
    return Total_loss.item(), x_grad

def objective_first(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu_first, u, coeffs, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L1 = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()

def objective_second(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu_second, u, COEFFS, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L1+L2 = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()


if __name__ == "__main__":
    print("Building operator for first step...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=devices[0])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11"])
    coeffs = wph_op.apply(Mixture, norm=norm, pbc=pbc).to("cpu")
    std_target = compute_std_L1(torch.from_numpy(Dust).to(0))
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print(f"Done! (in {time.time() - start_time}s)")
    
    ## Minimization
    
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = Mixture
    result = opt.minimize(objective_first, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params0)
    final_loss, s0_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    # Output
    
    # Reshaping
    s0_tilde = s0_tilde.reshape((M, N)).astype(np.float32)
    
    print(f"First step of denoising ended in {niter} iterations with optimizer message: {msg}")

    wph_op.to(devices[0])
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    coeffs_dust = wph_op.apply(Mixture, norm=norm, pbc=pbc).to("cpu")
    std_dust = compute_std_L1(torch.from_numpy(Dust).to(0))
    coeffs_CN = wph_op.apply(CIB_Noise, norm=norm, pbc=pbc).to("cpu")
    std_CN = compute_std_L1(torch.from_numpy(Dust*0).to(0))
    COEFFS = [coeffs_dust,coeffs_CN,std_dust,std_CN]
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print("Done! (in {:}s)".format(time.time() - start_time))
    
    ## Minimization
    
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = s0_tilde #Mixture
    result = opt.minimize(objective_second, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    ## Output
    
    # Reshaping
    s_tilde = s_tilde.reshape((M, N)).astype(np.float32)
    
    print("Denoising ended in {:} iterations with optimizer message: {:}".format(niter,msg))
    
    if output_filename is not None:
        np.save(output_filename, [Mixture,Dust,CIB_Noise,s_tilde,Mixture-s_tilde,s0_tilde,Mixture-s0_tilde])



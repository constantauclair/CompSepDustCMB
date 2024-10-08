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

#######
# INPUT PARAMETERS
#######

file_name="denoising_L12_SNR=0,5_true_noisy.npy"

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True
norm="auto"

alpha = 1

SNR = 0.5

optim_params = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

devices = [0,1] # List of GPUs to use

Mn = 100 # Number of noises per iteration

## Loading the data

Dust = np.load('../data/I_maps_v2_leo.npy')[0,0][::2,::2]

Noise = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[0].real

Noise_syn = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[1:Mn+1].real

## Normalizing the data

Dust = (Dust - np.mean(Dust)) / np.std(Dust)

Noise = (Noise - np.mean(Noise)) / np.std(Noise) / SNR / 2

Noise_syn = (Noise_syn - np.mean(Noise_syn)) / np.std(Noise_syn) / SNR / 2

Noise_model = Noise_syn[0]

Dust = Dust + Noise

Mixture = Dust + Noise

#######
# DENOISING
#######

def create_batch(n_maps, n, device, batch_size = 10):
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

def objective_per_gpu(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_tar_1 = coeffs_target[0].to(device_id)
    norm_map_1 = torch.from_numpy(coeffs_target[1]).to(device_id)
    
    coeffs_tar_2 = coeffs_target[2].to(device_id)
    norm_map_2 = torch.from_numpy(coeffs_target[3]).to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the 100 noise maps
    Noise_syn_ = torch.from_numpy(Noise_syn[work_list]).to(device)
    
    # Compute the L1
    wph_op.clear_normalization()
    wph_op.apply(norm_map_1, norm=norm, pbc=pbc)
    noise_batch = create_batch(len(Noise_syn_), Noise_syn_, device=device)
    loss_tot1 = torch.zeros(1)
    for i in range(noise_batch.shape[0]):
        u_noisy, nb_chunks = wph_op.preconfigure(u + noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_1[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot1 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
        
    # Compute the L2
    wph_op.clear_normalization()
    wph_op.apply(norm_map_2, norm=norm, pbc=pbc)
    loss_tot2 = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - u, pbc=pbc)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_2[indices]) ** 2) / Mn
        loss = loss*alpha
        loss.backward(retain_graph=True)
        loss_tot2 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    sys.stdout.flush() # Flush the standard output
    
    if device == 0:
        print("L1 =",loss_tot1.item())
        print("L2 =",loss_tot2.item())
    
    # Total loss
    Total_loss = loss_tot1 + loss_tot2
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del Noise_syn_, u # To free GPU memory
    
    return Total_loss.item(), x_grad

def objective(x):
    
    global eval_cnt
    print(f"Evaluation : {eval_cnt}")
    start_time = time.time()
    
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu, u, COEFFS, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print(f"L1 + L2 = {loss_tot} (computed in {time.time() - start_time}s)")
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()


if __name__ == "__main__":
    
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=devices[0])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    norm_1 = Dust
    norm_2 = Noise_model
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.apply(norm_1, norm=norm, pbc=pbc).to("cpu")
    coeffs_1 = wph_op.apply(Mixture, norm=norm, pbc=pbc).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply(norm_2, norm=norm, pbc=pbc).to("cpu")
    coeffs_2 = wph_op.apply(Noise_model, norm=norm, pbc=pbc).to("cpu")
    COEFFS = [coeffs_1,norm_1,coeffs_2,norm_2]
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print("Done! (in {:}s)".format(time.time() - start_time))
    
    ## Minimization
    
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = Mixture
    result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    ## Output
    
    # Reshaping
    Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
    
    print("Denoising ended in {:} iterations with optimizer message: {:}".format(niter,msg))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde])
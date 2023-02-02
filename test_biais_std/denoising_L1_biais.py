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

file_name="denoising_L1_biais.npy"

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True
norm="auto"

SNR = 1

n_step = 5
iter_per_step = 20

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 100 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

## Loading the data

Dust = np.load('../data/I_maps_v2_leo.npy')[0,0][::2,::2]

Noise = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[0].real

Noise_syn = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[1:Mn+1].real

## Normalizing the data

Dust = (Dust - np.mean(Dust)) / np.std(Dust)

Noise = (Noise - np.mean(Noise)) / np.std(Noise) / SNR

Noise_syn = (Noise_syn - np.mean(Noise_syn)) / np.std(Noise_syn) / SNR

Mixture = Dust + Noise

#######
# DENOISING
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
    
def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    x_curr = x.reshape((M, N))
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_target[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")

    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()

if __name__ == "__main__":
    
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    true_coeffs = wph_op.apply(Dust, norm=norm, pbc=pbc)
    
    ## Minimization
    
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step):
        # Initialization of the map
        if i == 0:
            x0 = torch.from_numpy(Mixture).to(device)
        else:
            x0 = Dust_tilde
        
        # Bias computation
        print("Computing target coeffs...")
        start_time = time.time()
        noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
        COEFFS_ = torch.zeros((n_batch,len(true_coeffs))).type(torch.double)
        for i in range(noise_batch.shape[0]):
            u_noisy, nb_chunks = wph_op.preconfigure(x0 + noise_batch[i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
                COEFFS_[i,indices] = torch.mean(coeffs_chunk,axis=0)
                del coeffs_chunk, indices
            sys.stdout.flush() # Flush the standard output
        
        mean_coeffs = torch.mean(COEFFS_,axis=0) # mean coeffs of u+n_i
        
        bias = mean_coeffs - wph_op.apply(x0, norm=norm, pbc=pbc) # mean coeffs of u+n_i - coeffs of u
            
        coeffs_target = wph_op.apply(x0, norm=norm, pbc=pbc) - bias # estimation of the unbiased coefficients
        print("Done ! (in {:}s)".format(time.time() - start_time))
        
        result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    ## Output
    
    # Reshaping
    Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
    
    print("Denoising ended in {:} iterations with optimizer message: {:}".format(niter,msg))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde])
        
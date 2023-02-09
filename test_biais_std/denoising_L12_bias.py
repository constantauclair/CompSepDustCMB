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

file_name="denoisings/denoising_L12_bias_s_n_norm.npy"

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True
norm="auto"

alpha = 1 # L2 factor

SNR = 1

n_step = 10
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

Noise_model = np.load('../data/BICEP_noise_QiU_217GHZ.npy')[Mn+1].real

## Normalizing the data

Dust = (Dust - np.mean(Dust)) / np.std(Dust)

Noise = (Noise - np.mean(Noise)) / np.std(Noise) / SNR

Noise_syn = (Noise_syn - np.mean(Noise_syn)) / np.std(Noise_syn) / SNR

Noise_model = (Noise_model - np.mean(Noise_model)) / np.std(Noise_model) / SNR

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
    
def compute_bias(x):
    print("Computing bias...")
    local_start_time = time.time()
    noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
    wph_op.clear_normalization()
    wph_op.apply(torch.from_numpy(norm1).to(device), norm=norm, pbc=pbc)
    coeffs_ref = wph_op.apply(x, norm=norm, pbc=pbc)
    bias = coeffs_ref * 0
    for i in range(noise_batch.shape[0]):
        u_noisy, nb_chunks = wph_op.preconfigure(x + noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            bias[indices] += torch.sum(coeffs_chunk - coeffs_ref[indices], axis=0) / Mn
            del coeffs_chunk, indices
        del u_noisy, nb_chunks
        sys.stdout.flush() # Flush the standard output
    print("Done ! (in {:}s)".format(time.time() - local_start_time))
    return bias

def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    x_curr = x.reshape((M, N))
    
    # Compute the loss 1
    wph_op.clear_normalization()
    wph_op.apply(torch.from_numpy(norm1).to(device), norm=norm, pbc=pbc)
    loss_tot1 = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_target[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot1 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
        
    # Compute the loss 2
    wph_op.clear_normalization()
    wph_op.apply(torch.from_numpy(norm2).to(device), norm=norm, pbc=pbc)
    loss_tot2 = torch.zeros(1)
    n_curr, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - x_curr, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(n_curr, i, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_noise[indices]) ** 2)
        loss = alpha * loss
        loss.backward(retain_graph=True)
        loss_tot2 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot1 + loss_tot2
    
    print(f"L = {loss_tot1.item()} + {loss_tot2.item()} = {loss_tot.item()} (computed in {time.time() - start_time}s)")

    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    norm1 = Dust
    norm2 = Noise_model
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    
    wph_op.apply(norm1, norm=norm, pbc=pbc)
    true_coeffs = wph_op.apply(Dust, norm=norm, pbc=pbc)
    wph_op.clear_normalization()
    
    wph_op.apply(norm2, norm=norm, pbc=pbc)
    coeffs_noise = wph_op.apply(Noise_model, norm=norm, pbc=pbc)
    wph_op.clear_normalization()
    
    ## Minimization
    
    eval_cnt = 0
    
    Dust_tilde = Mixture
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step):
        
        # Initialization of the map
        Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
        
        # Bias computation
        bias = compute_bias(Dust_tilde)
        
        # Coeffs target computation
        wph_op.clear_normalization()
        wph_op.apply(torch.from_numpy(norm1).to(device), norm=norm, pbc=pbc)
        coeffs_target = wph_op.apply(torch.from_numpy(Mixture), norm=norm, pbc=pbc) - bias # estimation of the unbiased coefficients
        
        # Minimization
        result = opt.minimize(objective, Dust_tilde.cpu().ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
        #result = opt.minimize(objective, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde])
        
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

file_name="denoisings/denoising_L1_bias_std_loss_per_coeffs.npy"

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True

SNR = 1

n_step1 = 1
iter_per_step1 = 50

n_step2 = 10
iter_per_step2 = 20

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

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

def compute_bias_std(x,norm):
    print("Computing bias and std...")
    local_start_time = time.time()
    noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
    coeffs_ref = wph_op.apply(x, norm=norm, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        u_noisy, nb_chunks = wph_op.preconfigure(x + noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk - coeffs_ref[indices]
            del coeffs_chunk, indices
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=0)
    std = torch.std(COEFFS,axis=0)
    print("Done ! (in {:}s)".format(time.time() - local_start_time))
    return bias, std

def compute_coeffs_bias_std(x,norm):
    print("Computing bias and std...")
    local_start_time = time.time()
    noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
    BIAS = []
    STD = []
    for i in range(len(wph_model)):
        wph = wph_op.apply(x, norm=norm, pbc=pbc, ret_wph_obj=True)
        coeffs_ref = wph.get_coeffs(wph_model[i])[0]
        coeffs_number = len(coeffs_ref)
        COEFFS = torch.zeros((Mn,coeffs_number))
        computed_noise = 0
        for j in range(noise_batch.shape[0]):
            this_batch_size = len(noise_batch[j])
            u_noisy = x + noise_batch[j]
            wph = wph_op.apply(u_noisy, norm=norm, pbc=pbc, ret_wph_obj=True)
            coeffs = wph.get_coeffs(wph_model[i])[0] - coeffs_ref
            COEFFS[computed_noise:computed_noise+this_batch_size] = torch.from_numpy(coeffs)
            computed_noise += this_batch_size
            del u_noisy, this_batch_size, coeffs
            sys.stdout.flush() # Flush the standard output
        BIAS.append(torch.mean(COEFFS,axis=0))
        STD.append(torch.std(COEFFS,axis=0))
    print("Done ! (in {:}s)".format(time.time() - local_start_time))
    return BIAS, STD

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    x_curr = x.reshape((M, N))
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=None, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs( (coeffs_chunk - coeffs_target[indices]) / std[indices] ) ** 2)
        loss = loss / len(indices)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")

    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    x_curr = x.reshape((M, N))
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    x_curr = torch.from_numpy(x_curr).requires_grad_(True)
    #x_curr, _ = wph_op.preconfigure(x_curr, requires_grad=True, pbc=pbc)
    wph = wph_op.apply(x_curr, norm='auto', pbc=pbc, ret_wph_obj=True)
    for i in range(len(wph_model)):
        coeffs = torch.from_numpy(wph.get_coeffs(wph_model[i])[0])
        loss = torch.sum(torch.abs( (coeffs - coeffs_target[i]) / std[i] ) ** 2)
        loss = loss / (len(coeffs) * len(wph_model))
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs, loss
    
    # Reshape the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")

    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first step of minimization (only S11)...")
    
    eval_cnt = 0
    
    Dust_tilde0 = Mixture
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias, std = compute_bias_std(Dust_tilde0,None)
        
        # Coeffs target computation
        coeffs_target = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc) - bias # estimation of the unbiased coefficients
        
        # Minimization
        #result = opt.minimize(objective1, Dust_tilde0.cpu().ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        result = opt.minimize(objective1, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((M, N)).astype(np.float32)
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    wph_op.clear_normalization()
    wph_model = ["S11","S00","S01","Cphase","C01","C00","L"]
    wph_op.load_model(wph_model)
    
    wph_op.apply(Dust_tilde0,norm='auto',pbc=pbc)
    
    Dust_tilde = Dust_tilde0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        # Initialization of the map
        Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
        
        # Bias computation
        bias, std = compute_coeffs_bias_std(Dust_tilde,'auto')
        
        # Coeffs target computation
        coeffs_target = []
        for j in range(len(wph_model)):
            wph = wph_op.apply(torch.from_numpy(Mixture), norm='auto', pbc=pbc, ret_wph_obj=True)
            coeffs_target.append(torch.from_numpy(wph.get_coeffs(wph_model[j])[0]) - bias[j])
        
        # Minimization
        #result = opt.minimize(objective2, Dust_tilde.cpu().ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        result = opt.minimize(objective2, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde])
        
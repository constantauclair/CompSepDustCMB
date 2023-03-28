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

n_iteration = 10

M, N = 256, 256
J = 6
L = 4
dn = 3
pbc = True

SNR = 0.5

file_names = []
for i in range(n_iteration):
    file_names.append("denoisings/iterative_denoising_final_L12_SNR=0,5_"+str(i+1)+"_of_"+str(n_iteration)+"_dn=3_ni=10.npy")

n_step1 = 3
iter_per_step1 = 50

n_step2 = 5
iter_per_step2 = 50

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

factor = 2*(n_iteration-np.arange(n_iteration))-1

Mn = 200 # Number of noises per iteration
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

noise_batch = create_batch(Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size) / n_iteration

def compute_bias_std(x,iteration):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        u_noisy, nb_chunks = wph_op.preconfigure(x + np.sqrt(factor[iteration])*noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk - coeffs_ref[indices]
            del coeffs_chunk, indices
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=0)
    std = torch.std(COEFFS,axis=0)
    return bias, std

def compute_complex_bias_std(x,iteration):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        u_noisy, nb_chunks = wph_op.preconfigure(x + np.sqrt(factor[iteration])*noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk - coeffs_ref[indices]
            del coeffs_chunk, indices
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = [torch.mean(torch.real(COEFFS),axis=0),torch.mean(torch.imag(COEFFS),axis=0)]
    std = [torch.std(torch.real(COEFFS),axis=0),torch.std(torch.imag(COEFFS),axis=0)]
    return bias, std

def compute_complex_bias_std_noise(x,fac):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[0]):
        this_batch_size = len(noise_batch[i])
        batch_COEFFS = torch.zeros((this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        u_noisy, nb_chunks = wph_op.preconfigure(x + fac*noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk - coeffs_ref[indices]
            del coeffs_chunk, indices
        COEFFS[computed_noise:computed_noise+this_batch_size] = batch_COEFFS
        computed_noise += this_batch_size
        del u_noisy, nb_chunks, batch_COEFFS, this_batch_size
        sys.stdout.flush() # Flush the standard output
    bias = [torch.mean(torch.real(COEFFS),axis=0),torch.mean(torch.imag(COEFFS),axis=0)]
    std = [torch.std(torch.real(COEFFS),axis=0),torch.std(torch.imag(COEFFS),axis=0)]
    return bias, std

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((M, N))
    
    # Compute the loss 1
    loss_tot = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs( (coeffs_chunk - coeffs_target[indices]) / std[indices] ) ** 2)
        loss = loss / len(coeffs_target)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    print("L = "+str(round(loss_tot.item(),3))+" (computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    
    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss 1
    loss_tot_1_real = torch.zeros(1)
    loss_tot_1_imag = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - coeffs_target[0][indices]) / std[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - coeffs_target[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / len(indices) #real_coeffs_number_dust
        loss_imag = loss_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_dust
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        loss_tot_1_real += loss_real.detach().cpu()
        loss_tot_1_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
        
    # Compute the loss 2
    loss_tot_2_real = torch.zeros(1)
    loss_tot_2_imag = torch.zeros(1)
    u_bis, nb_chunks = wph_op.preconfigure(torch.from_numpy(Iteration_map).to(device) - u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_bis, i, norm=None, ret_indices=True, pbc=pbc)
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - mean_noise[0][indices]) / std_noise[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_noise[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - mean_noise[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / real_coeffs_number_noise
        loss_imag = loss_imag / imag_coeffs_number_noise
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        loss_tot_2_real += loss_real.detach().cpu()
        loss_tot_2_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
        
    # Compute the loss 2 iterative
    loss_tot_2_iterative_real = torch.zeros(1)
    loss_tot_2_iterative_imag = torch.zeros(1)
    u_ter, nb_chunks = wph_op.preconfigure(torch.from_numpy(Iteration_map).to(device) - u + torch.from_numpy(Removed_Noise).to(device), requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_ter, i, norm=None, ret_indices=True, pbc=pbc)
        loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - iterative_mean_noise[0][indices]) / iterative_std_noise[0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / iterative_std_noise[1][indices],nan=0)
        loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - iterative_mean_noise[1][indices]) * kept_coeffs ) ** 2)
        loss_real = loss_real / real_coeffs_number_noise
        loss_imag = loss_imag / imag_coeffs_number_noise
        loss_real.backward(retain_graph=True)
        loss_imag.backward(retain_graph=True)
        loss_tot_2_iterative_real += loss_real.detach().cpu()
        loss_tot_2_iterative_imag += loss_imag.detach().cpu()
        del coeffs_chunk, indices, loss_real, loss_imag
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_1_real + loss_tot_1_imag + loss_tot_2_real + loss_tot_2_imag + loss_tot_2_iterative_real + loss_tot_2_iterative_imag
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L1 real = "+str(round(loss_tot_1_real.item(),3)))
    print("L1 imag = "+str(round(loss_tot_1_imag.item(),3)))
    print("L2 real = "+str(round(loss_tot_2_real.item(),3)))
    print("L2 imag = "+str(round(loss_tot_2_imag.item(),3)))
    print("L2 iterative real = "+str(round(loss_tot_2_iterative_real.item(),3)))
    print("L2 iterative imag = "+str(round(loss_tot_2_iterative_imag.item(),3)))
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
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    Iteration_map0 = Mixture
    
    Iteration_map = Iteration_map0
    
    Removed_Noise = np.zeros(np.shape(Noise))
    
    for iteration in range(n_iteration):
        
        print("Starting iteration "+str(iteration+1)+"...")
        iteration_start_time = time.time()
        
        ##########################################################################
    
        ## First minimization
        print("Starting first step of minimization (only S11)...")
        wph_op.load_model(["S11"])
        
        eval_cnt = 0
        
        Dust_tilde0 = Iteration_map
        
        # We perform a minimization of the objective function, using the noisy map as the initial map
        for i in range(n_step1):
            
            # Initialization of the map
            Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
            
            # Bias computation
            bias, std = compute_bias_std(Dust_tilde0,iteration)
            
            # Coeffs target computation
            coeffs_target = wph_op.apply(torch.from_numpy(Iteration_map), norm=None, pbc=pbc) - bias # estimation of the unbiased coefficients
            
            # Minimization
            result = opt.minimize(objective1, torch.from_numpy(Iteration_map).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
            final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
            
            # Reshaping
            Dust_tilde0 = Dust_tilde0.reshape((M, N)).astype(np.float32)
            
        ## Second minimization
        print("Starting second step of minimization (all coeffs)...")
        
        eval_cnt = 0
        
        # Identification of the irrelevant imaginary parts of the coeffs
        wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
        #wph_op.load_model(["S11","S00","S01","L"])
        wph_op.clear_normalization()
        coeffs_imag_dust = torch.imag(wph_op.apply(Dust_tilde0,norm=None,pbc=pbc))
        relevant_imaginary_coeffs_L1 = torch.where(torch.abs(coeffs_imag_dust) > 1e-6,1,0)
        coeffs_imag_noise = torch.imag(wph_op.apply(Noise,norm=None,pbc=pbc))
        relevant_imaginary_coeffs_L2 = torch.where(torch.abs(coeffs_imag_noise) > 1e-6,1,0)
        
        # Computation of the coeffs and std
        bias, std = compute_complex_bias_std(torch.from_numpy(Dust_tilde0).to(device),iteration)
        mean_noise, std_noise = compute_complex_bias_std_noise(torch.from_numpy(Dust_tilde0*0).to(device),1)
        iterative_mean_noise, iterative_std_noise = compute_complex_bias_std_noise(torch.from_numpy(Dust_tilde0*0).to(device),iteration+1)
        
        # Compute the number of coeffs
        real_coeffs_number_dust = len(torch.real(wph_op.apply(torch.from_numpy(Dust_tilde0).to(device),norm=None,pbc=pbc)))
        kept_coeffs_dust = torch.nan_to_num(relevant_imaginary_coeffs_L1 / std[1],nan=0)
        imag_coeffs_number_dust = torch.where(torch.sum(torch.where(kept_coeffs_dust>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_dust>0,1,0))).item()
        real_coeffs_number_noise = len(torch.real(wph_op.apply(torch.from_numpy(Noise).to(device),norm=None,pbc=pbc)))
        kept_coeffs_noise = torch.nan_to_num(relevant_imaginary_coeffs_L2 / std_noise[1],nan=0)
        imag_coeffs_number_noise = torch.where(torch.sum(torch.where(kept_coeffs_noise>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_noise>0,1,0))).item()
        
        Dust_tilde = Dust_tilde0
        
        # We perform a minimization of the objective function, using the noisy map as the initial map
        for i in range(n_step2):
            
            # Initialization of the map
            Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
            
            # Bias computation
            bias, std = compute_complex_bias_std(Dust_tilde,iteration)
            
            # Coeffs target computation
            coeffs_d = wph_op.apply(torch.from_numpy(Iteration_map), norm=None, pbc=pbc)
            coeffs_target = [torch.real(coeffs_d) - bias[0],torch.imag(coeffs_d) - bias[1]] # estimation of the unbiased coefficients
            
            # Minimization
            result = opt.minimize(objective2, torch.from_numpy(Dust_tilde0).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
            final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
            
            # Reshaping
            Dust_tilde = Dust_tilde.reshape((M, N)).astype(np.float32)
        
        print("Iteration "+str(iteration+1)+" done ! (in {:}s)".format(time.time() - iteration_start_time))
        
        Removed_Noise = Removed_Noise + Iteration_map - Dust_tilde
        
        np.save(file_names[iteration], [Mixture,Dust,Noise,Dust_tilde,Removed_Noise,Dust_tilde0,Iteration_map-Dust_tilde0])
        
        Iteration_map = Dust_tilde
        
        ##################################################################
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
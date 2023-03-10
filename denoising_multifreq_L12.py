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

n_freq = 2

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True

SNR = 1

file_name="denoising_multifreq_L12_SNR=1.npy"

n_step1 = 5
iter_per_step1 = 50

n_step2 = 10
iter_per_step2 = 50

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 200 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

## Loading the data

Dust_1 = np.load('data/I_maps_v2_leo.npy')[0,0][::2,::2]
Dust_2 = np.load('data/I_maps_v2_leo.npy')[0,1][::2,::2]

Noise_1 = np.load('data/BICEP_noise_QiU_217GHZ.npy')[0].real
Noise_2 = np.load('data/BICEP_noise_QiU_353GHZ.npy')[0].real

Noise_1_syn = np.load('data/BICEP_noise_QiU_217GHZ.npy')[1:Mn+1].real
Noise_2_syn = np.load('data/BICEP_noise_QiU_353GHZ.npy')[1:Mn+1].real

## Normalizing the data with respect to the first frequency

std_Dust_1 = np.std(Dust_1)
std_Noise_1 = np.std(Noise_1)

Noise_1 = Noise_1 / std_Noise_1 * std_Dust_1 / SNR
Noise_2 = Noise_2 / std_Noise_1 * std_Dust_1 / SNR

Noise_1_syn = Noise_1_syn / std_Noise_1 * std_Dust_1 / SNR
Noise_2_syn = Noise_2_syn / std_Noise_1 * std_Dust_1 / SNR

Mixture_1 = Dust_1 + Noise_1
Mixture_2 = Dust_2 + Noise_2

## Define final variables

print("Dust 1 std =",np.std(Dust_1))
print("Dust 2 std =",np.std(Dust_2))
print("Noise 1 std =",np.std(Noise_1))
print("Noise 2 std =",np.std(Noise_2))

Dust = np.array([Dust_1,Dust_2])
Noise = np.array([Noise_1,Noise_2])
Noise_syn = np.array([Noise_1_syn,Noise_2_syn])
Mixture = np.array([Mixture_1,Mixture_2])

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

def compute_bias_std(x):
    noise_batch = create_batch(n_freq, Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[1]):
        this_batch_size = len(noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + noise_batch[freq,i], pbc=pbc)
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

def compute_complex_bias_std(x):
    noise_batch = create_batch(n_freq, Mn, torch.from_numpy(Noise_syn).to(device), device=device, batch_size=batch_size)
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_noise = 0
    for i in range(noise_batch.shape[1]):
        this_batch_size = len(noise_batch[0,i])
        batch_COEFFS = torch.zeros((n_freq,this_batch_size,coeffs_number)).type(dtype=coeffs_ref.type())
        for freq in range(n_freq):
            u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + noise_batch[freq,i], pbc=pbc)
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

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_freq, M, N))
    
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
    u = x.reshape((n_freq, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss 1
    loss_tot_1_F1_real = torch.zeros(1)
    loss_tot_1_F2_real = torch.zeros(1)
    loss_tot_1_F1_imag = torch.zeros(1)
    loss_tot_1_F2_imag = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss F1
        print(std[1,0][indices])
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target[0,0][indices]) / std[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std[0,1][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target[0,1][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / len(indices) #real_coeffs_number_dust
        loss_F1_imag = loss_F1_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_dust
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target[1,0][indices]) / std[1,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / len(indices) #real_coeffs_number_dust
        loss_F2_imag = loss_F2_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_dust
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
    loss_tot_2_F1_real = torch.zeros(1)
    loss_tot_2_F2_real = torch.zeros(1)
    loss_tot_2_F1_imag = torch.zeros(1)
    loss_tot_2_F2_imag = torch.zeros(1)
    u_bis, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_bis, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - mean_noise[0,0][indices]) / std_noise[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_noise[0,1][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - mean_noise[0,1][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_noise
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_noise
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - mean_noise[1,0][indices]) / std_noise[1,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_noise[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - mean_noise[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_noise
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_noise
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_2_F1_real += loss_F1_real.detach().cpu()
        loss_tot_2_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_2_F2_real += loss_F2_real.detach().cpu()
        loss_tot_2_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_1_F1_real + loss_tot_1_F1_imag + loss_tot_1_F2_real + loss_tot_1_F2_imag + loss_tot_2_F1_real + loss_tot_2_F1_imag + loss_tot_2_F2_real + loss_tot_2_F2_imag
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L1 F1 real = "+str(round(loss_tot_1_F1_real.item(),3)))
    print("L1 F1 imag = "+str(round(loss_tot_1_F1_imag.item(),3)))
    print("L1 F2 real = "+str(round(loss_tot_1_F2_real.item(),3)))
    print("L1 F2 imag = "+str(round(loss_tot_1_F2_imag.item(),3)))
    print("L2 F1 real = "+str(round(loss_tot_2_F1_real.item(),3)))
    print("L2 F1 imag = "+str(round(loss_tot_2_F1_imag.item(),3)))
    print("L2 F2 real = "+str(round(loss_tot_2_F2_real.item(),3)))
    print("L2 F2 imag = "+str(round(loss_tot_2_F2_imag.item(),3)))
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
    print("Starting first step of minimization (only S11)...")
    
    eval_cnt = 0
    
    Dust_tilde0 = np.array([Mixture_1,Mixture_2])
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias, std = compute_bias_std(Dust_tilde0)
        mean_noise, std_noise = compute_bias_std(Dust_tilde0*0)
        
        # Coeffs target computation
        coeffs_target = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc) - bias # estimation of the unbiased coefficients
        
        # Minimization
        result = opt.minimize(objective1, torch.from_numpy(Mixture).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((n_freq, M, N)).astype(np.float32)
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Identification of the irrelevant imaginary parts of the coeffs
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    coeffs_imag_dust = torch.imag(wph_op.apply(Dust_tilde0[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L1 = torch.where(torch.abs(coeffs_imag_dust) > 1e-6,1,0)
    coeffs_imag_noise = torch.imag(wph_op.apply(Noise[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L2 = torch.where(torch.abs(coeffs_imag_noise) > 1e-6,1,0)
    
    # Computation of the coeffs and std
    bias, std = compute_complex_bias_std(torch.from_numpy(Dust_tilde0).to(device))
    mean_noise, std_noise = compute_complex_bias_std(torch.from_numpy(Dust_tilde0*0).to(device))
    
    # Compute the number of coeffs
    real_coeffs_number_dust = len(torch.real(wph_op.apply(torch.from_numpy(Dust_tilde0[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_dust = torch.nan_to_num(relevant_imaginary_coeffs_L1 / std[0,1],nan=0)
    imag_coeffs_number_dust = torch.where(torch.sum(torch.where(kept_coeffs_dust>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_dust>0,1,0))).item()
    real_coeffs_number_noise = len(torch.real(wph_op.apply(torch.from_numpy(Noise[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_noise = torch.nan_to_num(relevant_imaginary_coeffs_L2 / std_noise[0,1],nan=0)
    imag_coeffs_number_noise = torch.where(torch.sum(torch.where(kept_coeffs_noise>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_noise>0,1,0))).item()
    
    Dust_tilde = Dust_tilde0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        # Initialization of the map
        Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
        
        # Bias computation
        bias, std = compute_complex_bias_std(Dust_tilde)
        
        # Coeffs target computation
        coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias[:,0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias[:,1],dim=0))) # estimation of the unbiased coefficients
        
        # Minimization
        result = opt.minimize(objective2, torch.from_numpy(Dust_tilde0).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde = Dust_tilde.reshape((n_freq, M, N)).astype(np.float32)
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,Noise,Dust_tilde,Mixture-Dust_tilde,Dust_tilde0,Mixture-Dust_tilde0])
        
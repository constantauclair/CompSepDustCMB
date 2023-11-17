# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import argparse

'''
Starting from the result of the Bruno algorithm, this component separation algorithm aims to separate the polarized E dust emission from the CMB 
and noise contamination on Planck-like data. This is done at 353 GHz. The aim is to obtain the 
statistics of the E dust emission as well the TE correlation.
It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

The quantities involved are d (the total map), u (the optimized dust map), n (the noise + CMB map)
and T (the temperature map).

Loss terms:
    
# Dust 
L1 : (u + n_FM)= d_FM                      

# T correlation
L2 : (u + n_FM) x T = d_FM x T

# Noise + CMB  
L3 : (d_FM - u) = n_FM  

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('freq', type=int)
args = parser.parse_args()
freq = int(args.freq)

M, N = 512,512
J = 6#7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

file_name="separation_TE_correlation_"+str(freq)+"_Jean-Marc_post_Bruno.npy"

n_step = 5
iter_per_step = 50

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 50 # Number of noises per iteration
batch_size = 5
n_batch = int(Mn/batch_size)

###############################################################################
# DATA
###############################################################################

# Dust 
d_FM = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_'+str(freq)+'_FM.npy').astype(np.float32)

# CMB
c = np.load('data/IQU_Planck_data/TE correlation data/CMB_E_maps.npy').astype(np.float32)[:Mn]

# Noise
noise_set = np.load('data/IQU_Planck_data/TE correlation data/Noise_E_maps_'+str(freq)+'.npy').astype(np.float32)[:Mn]
n_FM = noise_set + c

# T map
T = np.load('data/IQU_Planck_data/TE correlation data/Planck_T_map_857.npy').astype(np.float32)

# Initial map
s_tilde0 = np.load('separation_TE_correlation_353_symcoeffs_512.npy').astype(np.float32)[2]

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

n_FM_batch = create_batch(torch.from_numpy(n_FM).to(device), device=device)

def compute_bias_std_L1(u_A, conta_A):
    coeffs_ref = wph_op.apply(u_A, norm=None, pbc=pbc)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(u_A + conta_A[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def compute_bias_std_L2(conta_A):
    coeffs_ref = wph_op.apply(conta_A[0,0], norm=None, pbc=pbc)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(conta_A[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def compute_bias_std_L3(u_A, conta_A):
    coeffs_ref = wph_op.apply([u_A,torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],torch.from_numpy(T).expand(conta_A[i].size()).to(device)], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    mask_real = torch.logical_and(torch.real(coeffs).to(device) > 1e-7, std[0].to(device) > 0)
    mask_imag = torch.logical_and(torch.imag(coeffs).to(device) > 1e-7, std[1].to(device) > 0)
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_loss(x,coeffs_target,std,mask,cross):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc, cross=cross)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc, cross=cross)
        loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() )
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    return loss_tot

###############################################################################
# OBJECTIVE FUNCTIONS
###############################################################################

def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    L1 = compute_loss(u,coeffs_target_L1,std_L1,mask_L1,cross=False)
    print(f"L1 = {round(L1.item(),3)}")
    L2 = compute_loss(torch.from_numpy(d_FM).to(device)-u,coeffs_target_L2,std_L2,mask_L2,cross=False)
    print(f"L2 = {round(L2.item(),3)}")
    L3 = compute_loss([u,torch.from_numpy(T).to(device)],coeffs_target_L3,std_L3,mask_L3,cross=True)
    print(f"L3 = {round(L3.item(),3)}")
    L = L1 + L2 + L3 # Define total loss 
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print(f"L = {round(L.item(),3)} (computed in {round(time.time() - start_time,3)} s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

###############################################################################
# MINIMIZATION
###############################################################################

if __name__ == "__main__":
    total_start_time = time.time()
    print("Starting component separation...")
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11","S00","S01","S10","Cphase","C01","C10","C00","L"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    print("Starting minimization...")
    eval_cnt = 0
    # Creating new variable
    s_tilde = s_tilde0
    # Computation of the coeffs, std and mask
    start_time_L2 = time.time()
    coeffs_target_L2, std_L2 = compute_bias_std_L2(n_FM_batch)
    mask_L2 = compute_mask(n_FM_batch[0,0],std_L2)
    print(f"L2 data computed in {time.time()-start_time_L2}")
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # Bias, coeffs target and mask computation
        start_time_L1 = time.time()
        bias_L1, std_L1 = compute_bias_std_L1(s_tilde, n_FM_batch)
        coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L1) - bias_L1[1],dim=0)))
        mask_L1 = compute_mask(s_tilde, std_L1)
        print(f"L1 data computed in {time.time()-start_time_L1}")
        start_time_L3 = time.time()
        bias_L3, std_L3 = compute_bias_std_L3(s_tilde, n_FM_batch)
        coeffs_L3 = wph_op.apply([torch.from_numpy(d_FM).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3) - bias_L3[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L3) - bias_L3[1],dim=0)))
        mask_L3 = compute_mask([s_tilde,torch.from_numpy(T).to(device)], std_L3, cross=True)
        print(f"L3 data computed in {time.time()-start_time_L3}")
        # Minimization
        result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, np.array([T,d_FM,s_tilde,d_FM-s_tilde,s_tilde0]))
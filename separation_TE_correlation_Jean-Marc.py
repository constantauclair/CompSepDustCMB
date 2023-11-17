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
This component separation algorithm aims to separate the polarized E dust emission from the CMB 
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
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

if freq == 100:
    fac_u0 = 0.5
if freq == 143:
    fac_u0 = 1
if freq == 217:
    fac_u0 = 2
if freq == 353:
    fac_u0 = 4

file_name="separation_TE_correlation_"+str(freq)+"_symcoeffs_512.npy"

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

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

n_FM_batch = create_batch(torch.from_numpy(n_FM).to(device), device=device)

def compute_std_L1(u_A, conta_A):
    coeffs_ref = wph_op.apply(u_A, norm=None, pbc=pbc, cross=False)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(u_A + conta_A[i], pbc=pbc, cross=False)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=False)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return std.to(device)

def compute_std_L2(u_A, conta_A):
    coeffs_ref = wph_op.apply([u_A,torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],torch.from_numpy(T).expand(conta_A[i].size()).to(device)], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return std.to(device)

def compute_mean_std_L3(conta_A):
    coeffs_ref = wph_op.apply(conta_A[0,0], norm=None, pbc=pbc, cross=False)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(conta_A[i], pbc=pbc, cross=False)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=False)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return mean.to(device), std.to(device)

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    thresh = 1e-5
    mask_real = torch.logical_and(torch.real(coeffs).to(device) > thresh, std[0].to(device) > 0)
    mask_imag = torch.logical_and(torch.imag(coeffs).to(device) > thresh, std[1].to(device) > 0)
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_L1(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(n_FM[j]).to(device), requires_grad=True, pbc=pbc, cross=False)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=False)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L2(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([x + torch.from_numpy(n_FM[j]).to(device),torch.from_numpy(T).to(device)], requires_grad=True, pbc=pbc, cross=True)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L3(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(d_FM).to(device) - x, requires_grad=True, pbc=pbc, cross=False)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=False)
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
    L1 = compute_L1(u,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L2(u,coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L3(u,coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    L = L1 + L2 + L3
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
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
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization...")
    eval_cnt = 0
    s_tilde0 = np.log(T) * fac_u0
    # L3
    coeffs_target_L3, std_L3 = compute_mean_std_L3(n_FM_batch)
    mask_L3 = compute_mask(n_FM_batch[0,0], std_L3)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        # L1
        std_L1 = compute_std_L1(s_tilde0, n_FM_batch)
        coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask(s_tilde0, std_L1)
        # L2
        std_L2 = compute_std_L2(s_tilde0, n_FM_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_FM).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask(s_tilde0, std_L2)
        # Minimization
        result = opt.minimize(objective, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second minimization...")
    eval_cnt = 0
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","S10","Cphase","C01","C10","C00","L"])
    wph_op.clear_normalization()
    # Creating new variable
    s_tilde = s_tilde0
    # L3
    coeffs_target_L3, std_L3 = compute_mean_std_L3(n_FM_batch)
    mask_L3 = compute_mask(n_FM_batch[0,0], std_L3)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1
        std_L1 = compute_std_L1(s_tilde, n_FM_batch)
        coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask(s_tilde, std_L1)
        # L2
        std_L2 = compute_std_L2(s_tilde, n_FM_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_FM).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask(s_tilde, std_L2)
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
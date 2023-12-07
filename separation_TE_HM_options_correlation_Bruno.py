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
L1 : (u + n_FM)= d_FM or (u + n_HM1) x (u + n_HM2) = d_HM1 x d_HM2                

# T correlation
L2 : (u + n_FM) x T = d_FM x T or nothing 

# Noise + CMB  
L3 : (d_FM - u) = n_FM  

u0 353 = 4 log(T)
u0 217 = 1 log(T)
u0 143 = 0.3 log(T)
u0 100 = 0.2 log(T)

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('freq', type=int)
parser.add_argument('crossTE', type=str)
parser.add_argument('crossHM', type=str)
args = parser.parse_args()
freq = int(args.freq)
E_or_TE = str(args.crossTE)
HM_or_FM = str(args.crossHM)

M, N = 768,768
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

fac_n_HM1 = 1.25
fac_n_HM2 = np.sqrt(4 - fac_n_HM1**2)

if freq == 100:
    fac_u0 = 0.2
if freq == 143:
    fac_u0 = 0.3
if freq == 217:
    fac_u0 = 1
if freq == 353:
    fac_u0 = 4

file_name="separation_correlation_"+str(freq)+"_symcoeffs_768_"+E_or_TE+"_"+HM_or_FM+".npy"

n_step = 5
iter_per_step = 50

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 50 # Number of noises per iteration
batch_size = 2
n_batch = int(Mn/batch_size)

wph_model = ["S11","S00","S01","Cphase","C01","C00","L"]
wph_model_cross = ["S11","S00","S01","S10","Cphase","C01","C10","C00","L"]

###############################################################################
# DATA
###############################################################################

# Dust 
d_FM = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_'+str(freq)+'_FM_768px.npy').astype(np.float32)
d_HM1 = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_'+str(freq)+'_HM1_768px.npy').astype(np.float32)
d_HM2 = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_'+str(freq)+'_HM2_768px.npy').astype(np.float32)

# CMB
c = np.load('data/IQU_Planck_data/TE correlation data/CMB_E_maps_768px.npy').astype(np.float32)[:Mn]

# Noise
noise_set = np.load('data/IQU_Planck_data/TE correlation data/Noise_E_maps_'+str(freq)+'_768px.npy').astype(np.float32)
n_FM = noise_set[:Mn] + c
n_HM1 = fac_n_HM1 * noise_set[:Mn] + c
n_HM2 = fac_n_HM2 * noise_set[Mn:2*Mn] + c

# T map
T = np.load('data/IQU_Planck_data/TE correlation data/Planck_T_map_857_768px.npy').astype(np.float32)

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

n_FM_batch = create_batch(torch.from_numpy(n_FM).to(device), device=device)
n_HM1_batch = create_batch(torch.from_numpy(n_HM1).to(device), device=device)
n_HM2_batch = create_batch(torch.from_numpy(n_HM2).to(device), device=device)

def compute_std_L1_FM(u_A, conta_A):
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

def compute_std_L1_HM(u_A, conta_A, conta_B):
    coeffs_ref = wph_op.apply([u_A,u_A], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],u_A + conta_B[i]], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
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

def compute_L1_FM(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(n_FM[j]).to(device), requires_grad=True, pbc=pbc, cross=False, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=False)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L1_HM(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([x + torch.from_numpy(n_HM1[j]).to(device),x + torch.from_numpy(n_HM2[j]).to(device)], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
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
        u_noisy, nb_chunks = wph_op.preconfigure([x + torch.from_numpy(n_FM[j]).to(device),torch.from_numpy(T).to(device)], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
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
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(d_FM).to(device) - x, requires_grad=True, pbc=pbc, cross=False, mem_chunk_factor=50, mem_chunk_factor_grad=80)
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

def objective_S11(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    if HM_or_FM == 'FM':
        L1 = compute_L1_FM(u,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    if HM_or_FM == 'HM':
        L1 = compute_L1_HM(u,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    if E_or_TE == 'TE':
        L2 = compute_L2(u,coeffs_target_L2,std_L2,mask_L2) # Compute L2
        print("L2 = "+str(round(L2.item(),3)))
    if E_or_TE == 'E':
        L2 = 0
    L3 = compute_L3(u,coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    L = L1 + L2 + L3
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    if HM_or_FM == 'FM':
        wph_op.load_model(wph_model)
        L1 = compute_L1_FM(u,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    if HM_or_FM == 'HM':
        wph_op.load_model(wph_model_cross)
        L1 = compute_L1_HM(u,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    if E_or_TE == 'TE':
        wph_op.load_model(wph_model_cross)
        L2 = compute_L2(u,coeffs_target_L2,std_L2,mask_L2) # Compute L2
        print("L2 = "+str(round(L2.item(),3)))
    if E_or_TE == 'E':
        L2 = 0
    wph_op.load_model(wph_model)
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
        if HM_or_FM == 'FM':
            std_L1 = compute_std_L1_FM(s_tilde0, n_FM_batch)
            coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask(s_tilde0, std_L1)
        if HM_or_FM == 'HM':
            std_L1 = compute_std_L1_HM(s_tilde0, n_HM1_batch, n_HM2_batch)
            coeffs_L1 = wph_op.apply([torch.from_numpy(d_HM1).to(device),torch.from_numpy(d_HM2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask([s_tilde0,s_tilde0], std_L1, cross=True)
        # L2
        if E_or_TE == 'TE':
            std_L2 = compute_std_L2(s_tilde0, n_FM_batch)
            coeffs_L2 = wph_op.apply([torch.from_numpy(d_FM).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
            mask_L2 = compute_mask([s_tilde0,torch.from_numpy(T).to(device)], std_L2, cross=True)
        # Minimization
        result = opt.minimize(objective_S11, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second minimization...")
    eval_cnt = 0
    # Initializing operator
    wph_op.clear_normalization()
    # Creating new variable
    s_tilde = s_tilde0
    # L3
    wph_op.load_model(wph_model)
    coeffs_target_L3, std_L3 = compute_mean_std_L3(n_FM_batch)
    mask_L3 = compute_mask(n_FM_batch[0,0], std_L3)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1
        if HM_or_FM == 'FM':
            wph_op.load_model(wph_model)
            std_L1 = compute_std_L1_FM(s_tilde, n_FM_batch)
            coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask(s_tilde, std_L1)
        if HM_or_FM == 'HM':
            wph_op.load_model(wph_model_cross)
            std_L1 = compute_std_L1_HM(s_tilde, n_HM1_batch, n_HM2_batch)
            coeffs_L1 = wph_op.apply([torch.from_numpy(d_HM1).to(device),torch.from_numpy(d_HM2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask([s_tilde,s_tilde], std_L1, cross=True)
        # L2
        if E_or_TE == 'TE':
            wph_op.load_model(wph_model_cross)
            std_L2 = compute_std_L2(s_tilde, n_FM_batch)
            coeffs_L2 = wph_op.apply([torch.from_numpy(d_FM).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
            mask_L2 = compute_mask([s_tilde,torch.from_numpy(T).to(device)], std_L2, cross=True)
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
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
This component separation algorithm aims to separate the polarized E and B dust emission from the CMB 
and noise contamination on Planck-like data. The aim is to obtain the statistics of the E and B dust 
emission as well the TE and TB correlation. It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

The quantities involved are d (the total map), u (the optimized dust map), n (the noise + CMB map) for E and B,
and T (the temperature map).

Loss terms:
    
# Dust EE, BB and EB
L1 : (u_E + n_E_HM1) x (u_E + n_E_HM2) = d_E_HM1 x d_E_HM2                
L2 : (u_B + n_B_HM1) x (u_B + n_B_HM2) = d_B_HM1 x d_B_HM2                
L3 : (u_E + n_E_FM) x (u_B + n_B_FM) = d_E_FM x d_B_FM 

# Dust TE and TB
L4 : (u_E + n_E_FM) x T = d_E_FM x T
L5 : (u_B + n_B_FM) x T = d_B_FM x T

# Noise + CMB EE and BB
L6 : (d_E_FM - u_E) = n_E_FM  
L7 : (d_B_FM - u_B) = n_B_FM  

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('freq', type=int) # 100, 143, 217 or 353
args = parser.parse_args()
freq = int(args.freq)

M, N = 768,768
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

file_name="separation_TEB_"+str(freq)+".npy"

Mn = 50
n_step = 5
iter_per_step = 50

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

batch_size = 5
n_batch = int(Mn/batch_size)

wph_model = ["S11","S00","S01","Cphase","C01","C00","L"]
wph_model_cross = ["S11","S00","S01","S10","Cphase","C01","C10","C00","L"]

###############################################################################
# DATA
###############################################################################

# Dust 
d_E_FM = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_E_'+str(freq)+'_FM_768px.npy').astype(np.float32)
d_E_HM1 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_E_'+str(freq)+'_HM1_768px.npy').astype(np.float32)
d_E_HM2 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_E_'+str(freq)+'_HM2_768px.npy').astype(np.float32)
d_B_FM = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_B_'+str(freq)+'_FM_768px.npy').astype(np.float32)
d_B_HM1 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_B_'+str(freq)+'_HM1_768px.npy').astype(np.float32)
d_B_HM2 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_B_'+str(freq)+'_HM2_768px.npy').astype(np.float32)

# CMB
c_E = np.load('data/IQU_Planck_data/TE_correlation_data/CMB/CMB_E_maps_'+str(freq)+'_768px.npy').astype(np.float32)[:Mn]
c_B = np.load('data/IQU_Planck_data/TE_correlation_data/CMB/CMB_B_maps_'+str(freq)+'_768px.npy').astype(np.float32)[:Mn]

# Noise
n_E_FM = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_E_'+str(freq)+'_FM_768px.npy').astype(np.float32)
n_E_HM1 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_E_'+str(freq)+'_HM1_768px.npy').astype(np.float32)
n_E_HM2 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_E_'+str(freq)+'_HM2_768px.npy').astype(np.float32)
n_B_FM = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_B_'+str(freq)+'_FM_768px.npy').astype(np.float32)
n_B_HM1 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_B_'+str(freq)+'_HM1_768px.npy').astype(np.float32)
n_B_HM2 = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_Noise_B_'+str(freq)+'_HM2_768px.npy').astype(np.float32)

# T map
T = np.load('data/IQU_Planck_data/TE_correlation_data/Sroll20/Sroll20_T_857_FM_768px.npy').astype(np.float32)

# CMB Noise combination
cn_E_FM = c_E + n_E_FM
cn_E_HM1 = c_E + n_E_HM1
cn_E_HM2 = c_E + n_E_HM2

cn_B_FM = c_B + n_B_FM
cn_B_HM1 = c_B + n_B_HM1
cn_B_HM2 = c_B + n_B_HM2

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

cn_E_FM_batch = create_batch(torch.from_numpy(cn_E_FM).to(device), device=device)
cn_E_HM1_batch = create_batch(torch.from_numpy(cn_E_HM1).to(device), device=device)
cn_E_HM2_batch = create_batch(torch.from_numpy(cn_E_HM2).to(device), device=device)

cn_B_FM_batch = create_batch(torch.from_numpy(cn_B_FM).to(device), device=device)
cn_B_HM1_batch = create_batch(torch.from_numpy(cn_B_HM1).to(device), device=device)
cn_B_HM2_batch = create_batch(torch.from_numpy(cn_B_HM2).to(device), device=device)

def compute_std_L123(u_A, u_B, conta_A, conta_B):
    coeffs_ref = wph_op.apply([u_A,u_B], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],u_B + conta_B[i]], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return std.to(device)

def compute_std_L45(u_A, conta_A):
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

def compute_mean_std_L67(conta_A):
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

def compute_L123(u_A,u_B,conta_A,conta_B,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([u_A + torch.from_numpy(conta_A[j]).to(device),u_B + torch.from_numpy(conta_B[j]).to(device)], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L45(u_A,conta_A,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([u_A + torch.from_numpy(conta_A[j]).to(device),torch.from_numpy(T).to(device)], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L67(u_A,obs_A,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(obs_A).to(device) - u_A, requires_grad=True, pbc=pbc, cross=False, mem_chunk_factor=50, mem_chunk_factor_grad=80)
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
    u = x.reshape((2, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_E = u[0]
    u_B = u[1]
    ######################
    L1 = compute_L123(u_E,u_E,torch.from_numpy(cn_E_HM1).to(device),torch.from_numpy(cn_E_HM2).to(device),coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_B,u_B,torch.from_numpy(cn_B_HM1).to(device),torch.from_numpy(cn_B_HM2).to(device),coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_E,u_B,torch.from_numpy(cn_E_FM).to(device),torch.from_numpy(cn_B_FM).to(device),coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    L4 = compute_L45(u_E,torch.from_numpy(cn_E_FM).to(device),coeffs_target_L4,std_L4,mask_L4) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_B,torch.from_numpy(cn_B_FM).to(device),coeffs_target_L5,std_L5,mask_L5) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    L6 = compute_L67(u_E,torch.from_numpy(d_E_FM).to(device),coeffs_target_L6,std_L6,mask_L6) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_B,torch.from_numpy(d_B_FM).to(device),coeffs_target_L7,std_L7,mask_L7) # Compute L7
    print("L7 = "+str(round(L7.item(),3)))
    ######################
    L = L1 + L2 + L3 + L4 + L5 + L6 + L7
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
    u = x.reshape((2, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_E = u[0]
    u_B = u[1]
    ######################
    wph_op.load_model(wph_model_cross)
    L1 = compute_L123(u_E,u_E,torch.from_numpy(cn_E_HM1).to(device),torch.from_numpy(cn_E_HM2).to(device),coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_B,u_B,torch.from_numpy(cn_B_HM1).to(device),torch.from_numpy(cn_B_HM2).to(device),coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_E,u_B,torch.from_numpy(cn_E_FM).to(device),torch.from_numpy(cn_B_FM).to(device),coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    wph_op.load_model(wph_model_cross)
    L4 = compute_L45(u_E,torch.from_numpy(cn_E_FM).to(device),coeffs_target_L4,std_L4,mask_L4) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_B,torch.from_numpy(cn_B_FM).to(device),coeffs_target_L5,std_L5,mask_L5) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    wph_op.load_model(wph_model)
    L6 = compute_L67(u_E,torch.from_numpy(d_E_FM).to(device),coeffs_target_L6,std_L6,mask_L6) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_B,torch.from_numpy(d_B_FM).to(device),coeffs_target_L7,std_L7,mask_L7) # Compute L7
    print("L7 = "+str(round(L7.item(),3)))
    ######################
    L = L1 + L2 + L3 + L4 + L5 + L6 + L7
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
    s_tilde0 = np.array([d_E_FM,d_B_FM])
    # L6
    coeffs_target_L6, std_L6 = compute_mean_std_L67(cn_E_FM_batch)
    mask_L6 = compute_mask(cn_E_FM_batch[0,0], std_L6)
    # L7
    coeffs_target_L7, std_L7 = compute_mean_std_L67(cn_B_FM_batch)
    mask_L7 = compute_mask(cn_B_FM_batch[0,0], std_L7)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        # L1
        std_L1 = compute_std_L123(s_tilde0[0], s_tilde0[0], cn_E_HM1_batch, cn_E_HM2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_E_HM1).to(device),torch.from_numpy(d_E_HM2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask([s_tilde0[0],s_tilde0[0]], std_L1, cross=True)
        #################### REPRENDRE ICI
        # L2
        std_L2 = compute_std_L2(s_tilde0, cn_FM_batch)
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
    coeffs_target_L3, std_L3 = compute_mean_std_L3(cn_FM_batch)
    mask_L3 = compute_mask(cn_FM_batch[0,0], std_L3)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1
        if HM_or_FM == 'FM':
            wph_op.load_model(wph_model)
            std_L1 = compute_std_L1_FM(s_tilde, cn_FM_batch)
            coeffs_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask(s_tilde, std_L1)
        if HM_or_FM == 'HM':
            wph_op.load_model(wph_model_cross)
            std_L1 = compute_std_L1_HM(s_tilde, cn_HM1_batch, cn_HM2_batch)
            coeffs_L1 = wph_op.apply([torch.from_numpy(d_HM1).to(device),torch.from_numpy(d_HM2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
            mask_L1 = compute_mask([s_tilde,s_tilde], std_L1, cross=True)
        # L2
        if E_or_TE == 'TE':
            wph_op.load_model(wph_model_cross)
            std_L2 = compute_std_L2(s_tilde, cn_FM_batch)
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
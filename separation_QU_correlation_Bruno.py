# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw

'''
This component separation algorithm aims to separate the polarized dust emission from the CMB 
and noise contamination on Planck-like data. This is done at 353 GHz. The aim is to obtain the 
statistics of the QU dust emission as well the TQ and TU correlation.
It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

The quantities involved are d (the total map), u (the optimized dust map), n (the noise + CMB map)
and T (the temperature map).

Loss terms:
    
# Dust 
L1 : (u_Q + n_Q) = d_Q    &    (u_U + n_U) = d_U                 

# Cross QU
L2 : (u_Q + n_Q) x (u_U + n_U) = d_Q x d_U

# T correlation
L3 : (u_Q + n_Q) x T = d_Q x T    &    (u_U + n_U) x T = d_U x T

# Noise + CMB  
L4 : (d_Q - u_Q) = n_Q    &    (d_U - u_U) = n_U  

u0 353 = 20 log(T)

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

M, N = 512, 512
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

file_name="separation_TQU_correlation_353.npy"

n_step = 5
iter_per_step = 50

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 50 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

###############################################################################
# DATA
###############################################################################

# Dust 
d_Q = np.load('data/IQU_Planck_data/multifrequency QU separation/Planck_IQU_map_353.npy')[1].astype(np.float32)
d_U = np.load('data/IQU_Planck_data/multifrequency QU separation/Planck_IQU_map_353.npy')[2].astype(np.float32)

# CMB
c_Q = np.load('data/IQU_Planck_data/multifrequency QU separation/CMB_IQU_maps.npy')[1].astype(np.float32)[:Mn]
c_U = np.load('data/IQU_Planck_data/multifrequency QU separation/CMB_IQU_maps.npy')[2].astype(np.float32)[:Mn]

# Noise
noise_Q = np.load('data/IQU_Planck_data/multifrequency QU separation/Noise_IQU_353.npy')[1].astype(np.float32)[:Mn]
noise_U = np.load('data/IQU_Planck_data/multifrequency QU separation/Noise_IQU_353.npy')[2].astype(np.float32)[:Mn]

# Total contamination
n_Q = noise_Q + c_Q
n_U = noise_U + c_U

# T map
T = np.load('data/IQU_Planck_data/multifrequency QU separation/Planck_T_map_857.npy').astype(np.float32)

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

n_Q_batch = create_batch(torch.from_numpy(n_Q).to(device), device=device)
n_U_batch = create_batch(torch.from_numpy(n_U).to(device), device=device)

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

def compute_std_L2(u_A, u_B, conta_A, conta_B):
    coeffs_ref = wph_op.apply([u_A, u_B], norm=None, pbc=pbc, cross=True)
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

def compute_std_L3(u_A, conta_A):
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

def compute_mean_std_L4(conta_A):
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

def compute_L1(x,n,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(n[j]).to(device), requires_grad=True, pbc=pbc, cross=False)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=False)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L2(x_Q,x_U,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([x_Q + torch.from_numpy(n_Q[j]).to(device),x_U + torch.from_numpy(n_U[j]).to(device)], requires_grad=True, pbc=pbc, cross=True)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L3(x,n,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([x + torch.from_numpy(n[j]).to(device),torch.from_numpy(T).to(device)], requires_grad=True, pbc=pbc, cross=True)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L4(x,d,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(d).to(device) - x, requires_grad=True, pbc=pbc, cross=False)
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
    u = x.reshape((2, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_Q = u[0]
    u_U = u[1]
    L1Q = compute_L1(u_Q,n_Q,coeffs_target_L1Q,std_L1Q,mask_L1Q) # Compute L1Q
    print("L1Q = "+str(round(L1Q.item(),3)))
    L1U = compute_L1(u_U,n_U,coeffs_target_L1U,std_L1U,mask_L1U) # Compute L1U
    print("L1U = "+str(round(L1U.item(),3)))
    L2 = compute_L2(u_Q,u_U,coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3Q = compute_L3(u_Q,n_Q,coeffs_target_L3Q,std_L3Q,mask_L3Q) # Compute L3Q
    print("L3Q = "+str(round(L3Q.item(),3)))
    L3U = compute_L3(u_U,n_U,coeffs_target_L3U,std_L3U,mask_L3U) # Compute L3U
    print("L3U = "+str(round(L3U.item(),3)))
    L4Q = compute_L4(u_Q,d_Q,coeffs_target_L4Q,std_L4Q,mask_L4Q) # Compute L4Q
    print("L4Q = "+str(round(L4Q.item(),3)))
    L4U = compute_L4(u_U,d_U,coeffs_target_L4U,std_L4U,mask_L4U) # Compute L4U
    print("L4U = "+str(round(L4U.item(),3)))
    L = L1Q + L1U + L2 + L3Q + L3U + L4Q + L4U
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
    s_tilde0 = np.array([20*np.log(T),20*np.log(T)])
    # L4
    coeffs_target_L4Q, std_L4Q = compute_mean_std_L4(n_Q_batch)
    mask_L4Q = compute_mask(n_Q_batch[0,0], std_L4Q)
    coeffs_target_L4U, std_L4U = compute_mean_std_L4(n_U_batch)
    mask_L4U = compute_mask(n_U_batch[0,0], std_L4U)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        # L1Q
        std_L1Q = compute_std_L1(s_tilde0[0], n_Q_batch)
        coeffs_L1Q = wph_op.apply(torch.from_numpy(d_Q).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1Q = torch.cat((torch.unsqueeze(torch.real(coeffs_L1Q),dim=0),torch.unsqueeze(torch.imag(coeffs_L1Q),dim=0)))
        mask_L1Q = compute_mask(s_tilde0[0], std_L1Q)
        # L1U
        std_L1U = compute_std_L1(s_tilde0[1], n_U_batch)
        coeffs_L1U = wph_op.apply(torch.from_numpy(d_U).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1U = torch.cat((torch.unsqueeze(torch.real(coeffs_L1U),dim=0),torch.unsqueeze(torch.imag(coeffs_L1U),dim=0)))
        mask_L1U = compute_mask(s_tilde0[1], std_L1U)
        # L2
        std_L2 = compute_std_L2(s_tilde0[0],s_tilde0[1], n_Q_batch, n_U_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_Q).to(device),torch.from_numpy(d_U).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask([s_tilde0[0],s_tilde0[1]], std_L2, cross=True)
        # L3Q
        std_L3Q = compute_std_L3(s_tilde0[0], n_Q_batch)
        coeffs_L3Q = wph_op.apply([torch.from_numpy(d_Q).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3Q = torch.cat((torch.unsqueeze(torch.real(coeffs_L3Q),dim=0),torch.unsqueeze(torch.imag(coeffs_L3Q),dim=0)))
        mask_L3Q = compute_mask([s_tilde0[0],torch.from_numpy(T).to(device)], std_L3Q, cross=True)
        # L3U
        std_L3U = compute_std_L3(s_tilde0[1], n_U_batch)
        coeffs_L3U = wph_op.apply([torch.from_numpy(d_U).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3U = torch.cat((torch.unsqueeze(torch.real(coeffs_L3U),dim=0),torch.unsqueeze(torch.imag(coeffs_L3U),dim=0)))
        mask_L3U = compute_mask([s_tilde0[1],torch.from_numpy(T).to(device)], std_L3U, cross=True)
        # Minimization
        result = opt.minimize(objective, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second minimization...")
    eval_cnt = 0
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    # Creating new variable
    s_tilde = s_tilde0
    # L4
    coeffs_target_L4Q, std_L4Q = compute_mean_std_L4(n_Q_batch)
    mask_L4Q = compute_mask(n_Q_batch[0,0], std_L4Q)
    coeffs_target_L4U, std_L4U = compute_mean_std_L4(n_U_batch)
    mask_L4U = compute_mask(n_U_batch[0,0], std_L4U)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1Q
        std_L1Q = compute_std_L1(s_tilde[0], n_Q_batch)
        coeffs_L1Q = wph_op.apply(torch.from_numpy(d_Q).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1Q = torch.cat((torch.unsqueeze(torch.real(coeffs_L1Q),dim=0),torch.unsqueeze(torch.imag(coeffs_L1Q),dim=0)))
        mask_L1Q = compute_mask(s_tilde[0], std_L1Q)
        # L1U
        std_L1U = compute_std_L1(s_tilde[1], n_U_batch)
        coeffs_L1U = wph_op.apply(torch.from_numpy(d_U).to(device), norm=None, pbc=pbc, cross=False)
        coeffs_target_L1U = torch.cat((torch.unsqueeze(torch.real(coeffs_L1U),dim=0),torch.unsqueeze(torch.imag(coeffs_L1U),dim=0)))
        mask_L1U = compute_mask(s_tilde[1], std_L1U)
        # L2
        std_L2 = compute_std_L2(s_tilde[0],s_tilde[1], n_Q_batch, n_U_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_Q).to(device),torch.from_numpy(d_U).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask([s_tilde[0],s_tilde[1]], std_L2, cross=True)
        # L3Q
        std_L3Q = compute_std_L3(s_tilde[0], n_Q_batch)
        coeffs_L3Q = wph_op.apply([torch.from_numpy(d_Q).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3Q = torch.cat((torch.unsqueeze(torch.real(coeffs_L3Q),dim=0),torch.unsqueeze(torch.imag(coeffs_L3Q),dim=0)))
        mask_L3Q = compute_mask([s_tilde[0],torch.from_numpy(T).to(device)], std_L3Q, cross=True)
        # L3U
        std_L3U = compute_std_L3(s_tilde[1], n_U_batch)
        coeffs_L3U = wph_op.apply([torch.from_numpy(d_U).to(device),torch.from_numpy(T).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3U = torch.cat((torch.unsqueeze(torch.real(coeffs_L3U),dim=0),torch.unsqueeze(torch.imag(coeffs_L3U),dim=0)))
        mask_L3U = compute_mask([s_tilde[1],torch.from_numpy(T).to(device)], std_L3U, cross=True)
        # Minimization
        result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, np.array([T,d_Q,d_U,s_tilde[0],s_tilde[1],d_Q-s_tilde[0],d_U-s_tilde[1],s_tilde0[0],s_tilde0[1]]))
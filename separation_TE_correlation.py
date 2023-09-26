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
L1 : (u + n_HM1) x (u + n_HM2) = d_HM1 x d_HM2      maybe L1bis : (u + n_FM) = d_FM ?

# Noise + CMB
L2 : (d_FM - u) = n_FM                              maybe L2bis : (d_HM1 - u) x (d_HM2 - u) = n_HM1 x n_HM2 ?

# T correlation
L3 : (u + n_FM) x T = d_FM x T

'''

###############################################################################
# INPUT PARAMETERS
###############################################################################

M, N = 512, 512
J = 7
L = 4
dn = 5
pbc = True
method = 'L-BFGS-B'

file_name="separation_TE_correlation.npy"

n_step1 = 5
iter_per_step1 = 50
    
n_step2 = 10
iter_per_step2 = 100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 50 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

###############################################################################
# DATA
###############################################################################

# Dust 
d_FM = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_353_FM.npy')
d_HM1 = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_353_HM1.npy')
d_HM2 = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_353_HM2.npy')

# CMB
c = np.load('data/IQU_Planck_data/TE correlation data/CMB_E_maps.npy')[:Mn]

# Noise
noise_set = np.load('data/IQU_Planck_data/TE correlation data/Noise_E_maps_353.npy')
n_FM = np.random.shuffle(noise_set)[:Mn] + c
n_HM1 = noise_set[:Mn] * np.sqrt(2) + c
n_HM2 = noise_set[Mn:2*Mn] * np.sqrt(2) + c

# T map
T = np.load('data/IQU_Planck_data/TE correlation data/Planck_T_map_857.npy')
    
# CMB
c = np.load('data/IQU_Planck_data/TE correlation data/CMB_E_maps.npy')[:Mn]

#######
# USEFUL FUNCTIONS
#######

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

n_FM_batch = create_batch(torch.from_numpy(n_FM).to(device), device=device)
n_HM1_batch = create_batch(torch.from_numpy(n_HM1).to(device), device=device)
n_HM2_batch = create_batch(torch.from_numpy(n_HM1).to(device), device=device)

def compute_bias_std_L1(u_A, u_B, conta_A, conta_B):
    coeffs_ref = wph_op.apply([u_A,u_B], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],u_B + conta_B[i]], pbc=pbc, cross=True)
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

def compute_bias_std_L2(conta_A, conta_B):
    coeffs_ref = wph_op.apply([conta_A[0,0],conta_B[0,0]], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([conta_A[i],conta_B[i]], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
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

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    L = compute_loss(u,coeffs_target_L1,std_L1,mask_L1,cross=False) # Compute the loss
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((3, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_217 = u[0]
    u_353 = u[1]
    u_CMB = u[2]
    L = 0 # Define total loss 
    # Compute the losses
    if '1' in losses:
        L1 = compute_loss(u_217,coeffs_target_L1,std_L1,mask_L1,cross=False)
        print(f"L1 = {round(L1.item(),3)}")
        L = L + L1
    if '2' in losses:
        L2 = compute_loss(u_353,coeffs_target_L2,std_L2,mask_L2,cross=False)
        print(f"L2 = {round(L2.item(),3)}")
        L = L + L2
    if '3' in losses:
        L3 = compute_loss(u_217-alpha*u_353,coeffs_target_L3,std_L3,mask_L3,cross=False)
        print(f"L3 = {round(L3.item(),3)}")
        L = L + L3
    if '4' in losses:
        L4 = compute_loss(u_CMB,coeffs_target_L4,std_L4,mask_L4,cross=False)
        print(f"L4 = {round(L4.item(),3)}")
        L = L + L4
    if '5' in losses:
        L5 = compute_loss([torch.from_numpy(d_217_1).to(device) - u_217 - u_CMB,torch.from_numpy(d_217_2).to(device) - u_217 - u_CMB],coeffs_target_L5,std_L5,mask_L5,cross=True)
        print(f"L5 = {round(L5.item(),3)}")
        L = L + L5
    if '6' in losses:
        L6 = compute_loss([torch.from_numpy(d_353_1).to(device) - u_353 - u_CMB,torch.from_numpy(d_353_2).to(device) - u_353 - u_CMB],coeffs_target_L6,std_L6,mask_L6,cross=True)
        print(f"L6 = {round(L6.item(),3)}")
        L = L + L6
    if '7' in losses:
        L7 = compute_loss([torch.from_numpy(d_217).to(device) - u_217 - u_CMB,torch.from_numpy(d_353).to(device) - u_353 - u_CMB],coeffs_target_L7,std_L7,mask_L7,cross=True)
        print(f"L7 = {round(L7.item(),3)}")
        L = L + L7
    if '8' in losses:
        L8 = compute_loss([u_217,torch.from_numpy(T_353).to(device)],coeffs_target_L8,std_L8,mask_L8,cross=True)
        print(f"L8 = {round(L8.item(),3)}")
        L = L + L8
    if '9' in losses:
        L9 = compute_loss([u_353,torch.from_numpy(T_353).to(device)],coeffs_target_L9,std_L9,mask_L9,cross=True)
        print(f"L9 = {round(L9.item(),3)}")
        L = L + L9
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print(f"L = {round(L.item(),3)} (computed in {round(time.time() - start_time,3)} s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

#######
# MINIMIZATION
#######

if __name__ == "__main__":
    total_start_time = time.time()
    print("Starting component separation on "+polar+" with L"+losses+" and a FBM of slope "+str(slope)+" as initial condition.")
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization (only S11)...")
    eval_cnt = 0
    Initial_condition = np.array([generate_fbm(d_217,n_217+CMB,slope),generate_fbm(d_353,n_353+CMB,slope)])
    s_tilde0 = Initial_condition
    for i in range(n_step1):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        s_217_tilde0 = s_tilde0[0]
        s_353_tilde0 = s_tilde0[1]
        # L1
        bias_L1, std_L1 = compute_bias_std_dust(s_217_tilde0, s_217_tilde0, CMB_batch + n_217_1_batch, CMB_batch + n_217_2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_217_1).to(device),torch.from_numpy(d_217_2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L1) - bias_L1[1],dim=0)))
        mask_L1 = compute_mask(s_217_tilde0, std_L1)
        # L2
        bias_L2, std_L2 = compute_bias_std_dust(s_353_tilde0, s_353_tilde0, CMB_batch + n_353_1_batch, CMB_batch + n_353_2_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_353_1).to(device),torch.from_numpy(d_353_2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2) - bias_L2[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L2) - bias_L2[1],dim=0)))
        mask_L2 = compute_mask(s_353_tilde0, std_L2)
        # Minimization
        result = opt.minimize(objective1, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params1)
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    eval_cnt = 0
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    # Creating new set of variables
    Initial_condition = np.array([s_tilde0[0],s_tilde0[1],np.random.normal(np.mean(CMB),np.std(CMB),size=(M,N))])
    # Computation of the coeffs, std and mask
    loss_data_mean_start_time = time.time()
    if '4' in losses:
        start_time_L4 = time.time()
        coeffs_target_L4, std_L4 = compute_bias_std_CMB()
        mask_L4 = compute_mask(torch.from_numpy(CMB).to(device),std_L4)
        print(f"L4 data computed in {time.time()-start_time_L4}")
    if '5' in losses:
        start_time_L5 = time.time()
        coeffs_target_L5, std_L5 = compute_bias_std_noise(n_217_1_batch,n_217_2_batch)
        mask_L5 = compute_mask([torch.from_numpy(n_217_1).to(device),torch.from_numpy(n_217_2).to(device)],std_L5,cross=True)
        print(f"L5 data computed in {time.time()-start_time_L5}")
    if '6' in losses:
        start_time_L6 = time.time()
        coeffs_target_L6, std_L6 = compute_bias_std_noise(n_353_1_batch,n_353_2_batch)
        mask_L6 = compute_mask([torch.from_numpy(n_353_1).to(device),torch.from_numpy(n_353_2).to(device)],std_L6,cross=True)
        print(f"L6 data computed in {time.time()-start_time_L6}")
    if '7' in losses:
        start_time_L7 = time.time()
        coeffs_target_L7, std_L7 = compute_bias_std_noise(n_217_batch,n_353_batch)
        mask_L7 = compute_mask([torch.from_numpy(n_217).to(device),torch.from_numpy(n_353).to(device)],std_L7,cross=True)
        print(f"L7 data computed in {time.time()-start_time_L7}")
    print(f"Loss data computed in {time.time() - loss_data_mean_start_time}")
    s_tilde = Initial_condition
    for i in range(n_step2):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        s_217_tilde = s_tilde[0]
        s_353_tilde = s_tilde[1]
        CMB_tilde = s_tilde[2]
        # Bias, coeffs target and mask computation
        loss_data_start_time = time.time()
        if '1' in losses:
            start_time_L1 = time.time()
            bias_L1, std_L1 = compute_bias_std_dust(s_217_tilde, s_217_tilde, CMB_batch + n_217_1_batch, CMB_batch + n_217_2_batch)
            coeffs_L1 = wph_op.apply([torch.from_numpy(d_217_1).to(device),torch.from_numpy(d_217_2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L1) - bias_L1[1],dim=0)))
            mask_L1 = compute_mask(s_217_tilde, std_L1)
            print(f"L1 data computed in {time.time()-start_time_L1}")
        if '2' in losses:
            start_time_L2 = time.time()
            bias_L2, std_L2 = compute_bias_std_dust(s_353_tilde, s_353_tilde, CMB_batch + n_353_1_batch, CMB_batch + n_353_2_batch)
            coeffs_L2 = wph_op.apply([torch.from_numpy(d_353_1).to(device),torch.from_numpy(d_353_2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2) - bias_L2[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L2) - bias_L2[1],dim=0)))
            mask_L2 = compute_mask(s_353_tilde, std_L2)
            print(f"L2 data computed in {time.time()-start_time_L2}")
        if '3' in losses:
            start_time_L3 = time.time()
            bias_L3, std_L3 = compute_bias_std_dust(s_217_tilde-alpha*s_353_tilde, s_217_tilde-alpha*s_353_tilde, (1-alpha)*CMB_batch + n_217_1_batch-alpha*n_353_1_batch, (1-alpha)*CMB_batch + n_217_2_batch-alpha*n_353_2_batch)
            coeffs_L3 = wph_op.apply([torch.from_numpy(d_217_1-alpha*d_353_1).to(device),torch.from_numpy(d_217_2-alpha*d_353_2).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3) - bias_L3[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L3) - bias_L3[1],dim=0)))
            mask_L3 = compute_mask(s_217_tilde-alpha*s_353_tilde, std_L3)
            print(f"L3 data computed in {time.time()-start_time_L3}")
        if '8' in losses:
            start_time_L8 = time.time()
            bias_L8, std_L8 = compute_bias_std_T(s_217_tilde, CMB_batch + n_217_batch)
            coeffs_L8 = wph_op.apply([torch.from_numpy(d_217).to(device),torch.from_numpy(T_353).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L8 = torch.cat((torch.unsqueeze(torch.real(coeffs_L8) - bias_L8[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L8) - bias_L8[1],dim=0)))
            mask_L8 = compute_mask([s_217_tilde,torch.from_numpy(T_353).to(device)], std_L8, cross=True)
            print(f"L8 data computed in {time.time()-start_time_L8}")
        if '9' in losses:
            start_time_L9 = time.time()
            bias_L9, std_L9 = compute_bias_std_T(s_353_tilde, CMB_batch + n_353_batch)
            coeffs_L9 = wph_op.apply([torch.from_numpy(d_353).to(device),torch.from_numpy(T_353).to(device)], norm=None, pbc=pbc, cross=True)
            coeffs_target_L9 = torch.cat((torch.unsqueeze(torch.real(coeffs_L9) - bias_L9[0],dim=0),torch.unsqueeze(torch.imag(coeffs_L9) - bias_L9[1],dim=0)))
            mask_L9 = compute_mask([s_353_tilde,torch.from_numpy(T_353).to(device)], std_L9, cross=True)
            print(f"L9 data computed in {time.time()-start_time_L9}")
        print(f"Loss data for era {str(i+1)} computed in {time.time() - loss_data_start_time}")
        # Minimization
        result = opt.minimize(objective2, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params2)
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((3, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, [np.array([d_217,d_353]),np.array([s_217,s_353]),np.array([CMB,CMB]),np.array([T_353,T_353]),np.array([n_217,n_353]),np.array([s_tilde[0],s_tilde[1]]),np.array([s_tilde[2],s_tilde[2]]),np.array([d_217-s_tilde[0]-s_tilde[2],d_353-s_tilde[1]-s_tilde[2]])])
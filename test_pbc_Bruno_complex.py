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

'''

###############################################################################
# INPUT PARAMETERS
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('pbc', type=int)
parser.add_argument('dn', type=int)
args = parser.parse_args()
pbc = bool(args.pbc)
dn = args.dn

M, N = 512, 512
J = 7
L = 4
method = 'L-BFGS-B'

file_name="test_pbc="+str(pbc)+"_dn="+str(dn)+"_logT_FM_Bruno_complex.npy"

print(file_name)

n_step1 = 5
iter_per_step1 = 50

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 50 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

###############################################################################
# DATA
###############################################################################

# Dust 
d_FM = np.load('data/IQU_Planck_data/TE correlation data/Planck_E_map_353_FM.npy').astype(np.float32)

# CMB
c = np.load('data/IQU_Planck_data/TE correlation data/CMB_E_maps.npy').astype(np.float32)[:Mn]

# Noise
noise_set = np.load('data/IQU_Planck_data/TE correlation data/Noise_E_maps_353.npy').astype(np.float32)
n_FM = noise_set[:Mn] + c

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

def compute_bias_std_L1(u_A, conta_A):
    coeffs_ref = wph_op.apply(u_A, norm=None, pbc=pbc, cross=False)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(u_A + conta_A[i], pbc=pbc, cross=False)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=False)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    bias = torch.mean(COEFFS,axis=0) #torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.std(COEFFS,axis=0) #torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    thresh = 1e-5
    mask = torch.abs(coeffs).to(device) > thresh
    return mask.to(device)

def compute_loss(x,coeffs_target,std,mask,cross):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(n_FM[j]).to(device), requires_grad=True, pbc=pbc, cross=cross)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=cross)
            loss = torch.sum( torch.abs( (coeffs_chunk[mask[indices]] - coeffs_target[indices][mask[indices]]) / std[indices][mask[indices]] ) ** 2) / mask.sum() / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

###############################################################################
# OBJECTIVE FUNCTIONS
###############################################################################

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
    print("Starting first minimization (only S11)...")
    eval_cnt = 0
    s_tilde = np.log(T) * 4 #np.random.normal(0,1,size=np.shape(d_FM)) #d_FM
    for i in range(n_step1):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1
        _, std_L1 = compute_bias_std_L1(s_tilde, n_FM_batch)
        print("Std =",std_L1)
        coeffs_target_L1 = wph_op.apply(torch.from_numpy(d_FM).to(device), norm=None, pbc=pbc, cross=False)
        print("Coeffs target =",coeffs_target_L1)
        mask_L1 = compute_mask(s_tilde, std_L1)
        print("Mask =",mask_L1)
        # Minimization
        result = opt.minimize(objective1, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params1)
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, np.array([T,d_FM,s_tilde,d_FM-s_tilde]))
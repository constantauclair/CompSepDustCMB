# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import argparse
sys.path.append('../Packages/')
import scattering as scat

'''
This component separation algorithm aims to separate the kSZ effect from the CMB. 
It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

Loss terms:
# Auto statistics
L1 : u_kSZ + CMB = d
L2 : d - u_kSZ = CMB

'''
#######
# INPUT PARAMETERS
#######


parser = argparse.ArgumentParser()
parser.add_argument('thresh', type=float)
parser.add_argument('contamination', type=int)
args = parser.parse_args()
threshold = args.thresh
conta = args.contamination

fac = 100

M, N = 500, 500
J = 7
L = 4
dn = 5
method = 'L-BFGS-B'

file_name="separation_kSZ_CMB_fac="+str(fac)+"_threshold="+str(int(100*threshold))+"_conta="+str(conta)+".npy"

n_step1 = 5
iter_per_step1 = 50
    
n_step2 = 10
iter_per_step2 = 100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 100 # Number of noises per iteration
batch_size = 5
n_batch = int(Mn/batch_size)

#######
# DATA
#######

kSZ = np.load('data/kSZ_CMB_data/kSZ.npy').astype(np.float32) * fac

CMB_syn = np.load('data/kSZ_CMB_data/CMB_syn.npy').astype(np.float32)

CMB = CMB_syn[conta]

Mixture = kSZ + CMB

#######
# USEFUL FUNCTIONS
#######

def compute_S11(st_calc,x):
    S11 = st_calc.scattering_cov_constant(x,only_S11=True)
    if len(np.shape(x)) == 2:
        return S11[0]
    if len(np.shape(x)) == 3:
        return S11
    
def compute_mask(st_calc,x,norm=True):
    return scat.compute_threshold_mask(st_calc.scattering_cov_constant(x,normalization=norm),threshold)
    
def compute_coeffs(st_calc,x,mask,norm=False,use_ref=False):
    coeffs = scat.threshold_coeffs(st_calc.scattering_cov_constant(x,normalization=norm,use_ref=use_ref),mask)
    if len(np.shape(x)) == 2:
        return coeffs[0]
    if len(np.shape(x)) == 3:
        return coeffs
    
def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

CMB_batch = create_batch(torch.from_numpy(CMB_syn).to(device), device=device)

def compute_bias_std_S11(u):
    coeffs_ref = compute_S11(st_calc,u)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        COEFFS[i*batch_size:(i+1)*batch_size] = compute_S11(st_calc, u + CMB_batch[i]) - coeffs_ref.type(dtype=ref_type)
    bias = torch.mean(COEFFS,0)
    std = torch.std(COEFFS,0)
    return bias.to(device), std.to(device)

def compute_bias_std_L1(u,mask):
    coeffs_ref = compute_coeffs(st_calc,u,mask)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        COEFFS[i*batch_size:(i+1)*batch_size] = compute_coeffs(st_calc,u + CMB_batch[i],mask) - coeffs_ref.type(dtype=ref_type)
    bias = torch.mean(COEFFS,0)
    std = torch.std(COEFFS,0)
    return bias.to(device), std.to(device)

def compute_mean_std_L2(mask):
    coeffs_ref = compute_coeffs(st_calc,torch.from_numpy(CMB).to(device),mask)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        COEFFS[i*batch_size:(i+1)*batch_size] = compute_coeffs(st_calc,CMB_batch[i],mask)
    mean = torch.mean(COEFFS,0)
    std = torch.std(COEFFS,0)
    return mean.to(device), std.to(device)

def compute_loss_S11(x,coeffs_target,std):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    loss_tot = torch.zeros(1)
    loss = torch.sum(torch.abs(( (compute_S11(st_calc,x) - coeffs_target)/std )**2))
    loss.backward(retain_graph=True)
    loss_tot += loss.detach().cpu()
    return loss_tot

def compute_loss(x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    loss = torch.sum(torch.abs(( (compute_coeffs(st_calc,x,mask) - coeffs_target)/std )**2))
    loss.backward(retain_graph=True)
    loss_tot += loss.detach().cpu()
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
    L1 = compute_loss_S11(u, coeffs_target, std) # Compute the loss
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print("L1 = "+str(round(L1.item(),3))+"(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    
    eval_cnt += 1
    return L1.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    u = x.reshape((M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    L1 = compute_loss(u, coeffs_target_L1, std_L1, mask_L1) # Compute the L1
    L2 = compute_loss(torch.from_numpy(Mixture).to(device) - u, coeffs_target_L2, std_L2, mask_L2) # Compute the L2
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    L = L1 + L2
    print("L = "+str(round(L.item(),3))+"(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L1 = "+str(round(L1.item(),3)))
    print("L2 = "+str(round(L2.item(),3)))
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

#######
# MINIMIZATION
#######

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Starting kSZ-CMB separation...")
    print("Building operator...")
    start_time = time.time()
    st_calc = scat.Scattering2d(M, N, J, L, device) 
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization (only S11)...")
    
    eval_cnt = 0
    
    Initial_condition = Mixture
    
    kSZ_tilde0 = Initial_condition
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        print("Starting era "+str(i+1)+"...")
        
        kSZ_tilde0 = torch.from_numpy(kSZ_tilde0).to(device) # Initialization of the map
        bias, std = compute_bias_std_S11(kSZ_tilde0) # Bias computation
        coeffs_d = compute_S11(st_calc,torch.from_numpy(Mixture).to(device))
        coeffs_target = coeffs_d - bias # Coeffs target computation
        result = opt.minimize(objective1, kSZ_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params1) # Minimization
        final_loss, kSZ_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        kSZ_tilde0 = kSZ_tilde0.reshape((M, N)).astype(np.float32) # Reshaping

        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    kSZ_tilde = kSZ_tilde0
    mask_L2 = compute_mask(st_calc,torch.from_numpy(CMB).to(device))
    coeffs_target_L2, std_L2 = compute_mean_std_L2(mask_L2)
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        print("Starting era "+str(i+1)+"...")
        
        kSZ_tilde = torch.from_numpy(kSZ_tilde).to(device) # Initialization of the map
        mask_L1 = compute_mask(st_calc, kSZ)
        bias_L1, std_L1 = compute_bias_std_L1(kSZ_tilde, mask_L1)
        coeffs_d = compute_coeffs(st_calc, torch.from_numpy(Mixture).to(device), mask_L1)
        coeffs_target_L1 = coeffs_d - bias_L1
        result = opt.minimize(objective2, kSZ_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params2) # Minimization
        final_loss, kSZ_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        kSZ_tilde = kSZ_tilde.reshape((M, N)).astype(np.float32) # Reshaping
        
        print("Era "+str(i+1)+" done !")
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,kSZ,CMB,kSZ_tilde,Mixture-kSZ_tilde])        
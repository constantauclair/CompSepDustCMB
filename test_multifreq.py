# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
from scipy.optimize import curve_fit
import scipy.optimize as opt
import pywph as pw
import argparse

'''
Loss terms:
    
# Dust polar
L1 : (u_217 + n_217_HM1) x (u_217 + n_217_HM2) = d_217_HM1 x d_217_HM2                
L2 : (u_353 + n_353_HM1) x (u_353 + n_353_HM2) = d_353_HM1 x d_353_HM2                
L3 : (u_217 + n_217_FM) x (u_353 + n_353_FM) = d_217_FM x d_353_FM 

# Dust crossI
L4 : (u_217 + n_217_FM) x I = d_217_FM x I
L5 : (u_353 + n_353_FM) x I = d_353_FM x I

# Noise + CMB
L6 : (d_217_FM - u_217) = n_217_FM  
L7 : (d_353_FM - u_353) = n_353_FM  

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

M, N = 768,768
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

file_name="separation_multifreq_Q_217_353_5steps_50iters_Mn=100.npy"

Mn = 100
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

print('Starting multifreq component separation GHz...')
print('Loading data...')

# Dust 

dust_maps = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_IQU_maps.npy').astype(np.float32)

d_217_FM = dust_maps[1,2,0]
d_217_HM1 = dust_maps[1,2,1]
d_217_HM2 = dust_maps[1,2,2]

d_353_FM = dust_maps[1,3,0]
d_353_HM1 = dust_maps[1,3,1]
d_353_HM2 = dust_maps[1,3,2]

del dust_maps

print('Dust loaded !')

# CMB

cmb_maps = np.load('data/IQU_Planck_data/Sroll20_data/CMB_IQU_maps.npy').astype(np.float32)

c_217 = cmb_maps[1,2,:Mn]

c_353 = cmb_maps[1,3,:Mn]

del cmb_maps

print('CMB loaded !')

# Noise

noise_217 = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_Noise_Q_217_maps.npy').astype(np.float32)

n_217_FM = noise_217[0,:Mn]
n_217_HM1 = noise_217[1,:Mn]
n_217_HM2 = noise_217[2,:Mn]

del noise_217

noise_353 = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_Noise_Q_353_maps.npy').astype(np.float32)

n_353_FM = noise_353[0,:Mn]
n_353_HM1 = noise_353[1,:Mn]
n_353_HM2 = noise_353[2,:Mn]

del noise_353

print('Noise loaded !')

# I map
I = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_I_857_FM_768px.npy').astype(np.float32)

# CMB Noise combination
cn_217_FM = c_217 + n_217_FM
cn_217_HM1 = c_217 + n_217_HM1
cn_217_HM2 = c_217 + n_217_HM2

cn_353_FM = c_353 + n_353_FM
cn_353_HM1 = c_353 + n_353_HM1
cn_353_HM2 = c_353 + n_353_HM2

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

print('Creating batches...')

cn_217_FM_batch = create_batch(torch.from_numpy(cn_217_FM).to(device), device=device)
cn_217_HM1_batch = create_batch(torch.from_numpy(cn_217_HM1).to(device), device=device)
cn_217_HM2_batch = create_batch(torch.from_numpy(cn_217_HM2).to(device), device=device)

cn_353_FM_batch = create_batch(torch.from_numpy(cn_353_FM).to(device), device=device)
cn_353_HM1_batch = create_batch(torch.from_numpy(cn_353_HM1).to(device), device=device)
cn_353_HM2_batch = create_batch(torch.from_numpy(cn_353_HM2).to(device), device=device)

print('Batches computed !')

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
    coeffs_ref = wph_op.apply([u_A,torch.from_numpy(I).to(device)], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],torch.from_numpy(I).expand(conta_A[i].size()).to(device)], pbc=pbc, cross=True)
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

def get_thresh(coeffs):
    coeffs_for_hist = np.abs(coeffs.cpu().numpy().flatten())
    non_zero_coeffs_for_hist = coeffs_for_hist[np.where(coeffs_for_hist>0)]
    hist, bins_edges = np.histogram(np.log10(non_zero_coeffs_for_hist),bins=100,density=True)
    bins = (bins_edges[:-1] + bins_edges[1:]) / 2
    x = bins
    y = hist
    def func(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
        y = amp1 * np.exp( -((x - mu1)/sigma1)**2) + amp2 * np.exp( -((x - mu2)/sigma2)**2)
        return y
    guess = [x[0]+(x[-1]-x[0])/4, 1, 0.3, x[0]+3*(x[-1]-x[0])/4, 1, 0.3]
    popt, pcov = curve_fit(func, x, y, p0=guess)
    thresh = 10**((popt[0]+popt[3])/2)
    return thresh

def compute_mask_S11(x,cross=False):
    if cross:
        wph_op.load_model(wph_model_cross)
    if not cross:
        wph_op.load_model(wph_model)
    full_coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    thresh = get_thresh(full_coeffs)
    wph_op.load_model(['S11'])
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    mask_real = torch.abs(torch.real(coeffs)).to(device) > thresh
    mask_imag = torch.abs(torch.imag(coeffs)).to(device) > thresh
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    thresh = get_thresh(coeffs)
    mask_real = torch.logical_and(torch.abs(torch.real(coeffs)).to(device) > thresh, std[0].to(device) > 0)
    mask_imag = torch.logical_and(torch.abs(torch.imag(coeffs)).to(device) > thresh, std[1].to(device) > 0)
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
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
        u_noisy, nb_chunks = wph_op.preconfigure([u_A + torch.from_numpy(conta_A[j]).to(device),torch.from_numpy(I).to(device)], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
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
    u_217 = u[0]
    u_353 = u[1]
    ######################
    L1 = compute_L123(u_217,u_217,cn_217_HM1,cn_217_HM2,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_353,u_353,cn_353_HM1,cn_353_HM2,coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_217,u_353,cn_217_FM,cn_353_FM,coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    L4 = compute_L45(u_217,cn_217_FM,coeffs_target_L4,std_L4,mask_L4) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_353,cn_353_FM,coeffs_target_L5,std_L5,mask_L5) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    L6 = compute_L67(u_217,d_217_FM,coeffs_target_L6,std_L6,mask_L6) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_353,d_353_FM,coeffs_target_L7,std_L7,mask_L7) # Compute L7
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
    u_217 = u[0]
    u_353 = u[1]
    ######################
    wph_op.load_model(wph_model_cross)
    L1 = compute_L123(u_217,u_217,cn_217_HM1,cn_217_HM2,coeffs_target_L1,std_L1,mask_L1) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_353,u_353,cn_353_HM1,cn_353_HM2,coeffs_target_L2,std_L2,mask_L2) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_217,u_353,cn_217_FM,cn_353_FM,coeffs_target_L3,std_L3,mask_L3) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    wph_op.load_model(wph_model_cross)
    L4 = compute_L45(u_217,cn_217_FM,coeffs_target_L4,std_L4,mask_L4) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_353,cn_353_FM,coeffs_target_L5,std_L5,mask_L5) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    wph_op.load_model(wph_model)
    L6 = compute_L67(u_217,d_217_FM,coeffs_target_L6,std_L6,mask_L6) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_353,d_353_FM,coeffs_target_L7,std_L7,mask_L7) # Compute L7
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
    s_tilde0 = np.array([d_217_FM,d_353_FM])
    # L6
    print('Preparing L6...')
    coeffs_target_L6, std_L6 = compute_mean_std_L67(cn_217_FM_batch)
    mask_L6 = compute_mask_S11(cn_217_FM_batch[0,0])
    print('L6 prepared !')
    # L7
    print('Preparing L7...')
    coeffs_target_L7, std_L7 = compute_mean_std_L67(cn_353_FM_batch)
    mask_L7 = compute_mask_S11(cn_353_FM_batch[0,0])
    print('L7 prepared !')
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        # L1
        print('Preparing L1...')
        std_L1 = compute_std_L123(s_tilde0[0], s_tilde0[0], cn_217_HM1_batch, cn_217_HM2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_217_HM1).to(device),torch.from_numpy(d_217_HM2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask_S11([s_tilde0[0],s_tilde0[0]], cross=True)
        print('L1 prepared !')
        # L2
        print('Preparing L2...')
        std_L2 = compute_std_L123(s_tilde0[1], s_tilde0[1], cn_353_HM1_batch, cn_353_HM2_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_353_HM1).to(device),torch.from_numpy(d_353_HM2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask_S11([s_tilde0[1],s_tilde0[1]], cross=True)
        print('L2 prepared !')
        # L3
        print('Preparing L3..')
        std_L3 = compute_std_L123(s_tilde0[0], s_tilde0[1], cn_217_FM_batch, cn_353_FM_batch)
        coeffs_L3 = wph_op.apply([torch.from_numpy(d_217_FM).to(device),torch.from_numpy(d_353_FM).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3),dim=0),torch.unsqueeze(torch.imag(coeffs_L3),dim=0)))
        mask_L3 = compute_mask_S11([s_tilde0[0],s_tilde0[1]], cross=True)
        print('L3 prepared !')
        # L4
        print('Preparing L4...')
        std_L4 = compute_std_L45(s_tilde0[0], cn_217_FM_batch)
        coeffs_L4 = wph_op.apply([torch.from_numpy(d_217_FM).to(device),torch.from_numpy(I).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_L4),dim=0),torch.unsqueeze(torch.imag(coeffs_L4),dim=0)))
        mask_L4 = compute_mask_S11([s_tilde0[0],torch.from_numpy(I).to(device)], cross=True)
        print('L4 prepared !')
        # L5
        print('Preparing L5...')
        std_L5 = compute_std_L45(s_tilde0[1], cn_353_FM_batch)
        coeffs_L5 = wph_op.apply([torch.from_numpy(d_353_FM).to(device),torch.from_numpy(I).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L5 = torch.cat((torch.unsqueeze(torch.real(coeffs_L5),dim=0),torch.unsqueeze(torch.imag(coeffs_L5),dim=0)))
        mask_L5 = compute_mask_S11([s_tilde0[1],torch.from_numpy(I).to(device)], cross=True)
        print('L5 prepared !')
        # Minimization
        print('Beginning optimization...')
        result = opt.minimize(objective_S11, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print('')
    print('')
    print('')
    print("Starting second minimization...")
    eval_cnt = 0
    s_tilde = s_tilde0
    # L6
    print('Preparing L6...')
    wph_op.load_model(wph_model)
    coeffs_target_L6, std_L6 = compute_mean_std_L67(cn_217_FM_batch)
    mask_L6 = compute_mask(cn_217_FM_batch[0,0], std_L6)
    print('L6 prepared !')
    # L7
    print('Preparing L7...')
    wph_op.load_model(wph_model)
    coeffs_target_L7, std_L7 = compute_mean_std_L67(cn_353_FM_batch)
    mask_L7 = compute_mask(cn_353_FM_batch[0,0], std_L7)
    print('L7 prepared !')
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        # L1
        print('Preparing L1...')
        wph_op.load_model(wph_model_cross)
        std_L1 = compute_std_L123(s_tilde[0], s_tilde[0], cn_217_HM1_batch, cn_217_HM2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_217_HM1).to(device),torch.from_numpy(d_217_HM2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask([s_tilde[0],s_tilde[0]], std_L1, cross=True)
        print('L1 prepared !')
        # L2
        print('Preparing L2...')
        wph_op.load_model(wph_model_cross)
        std_L2 = compute_std_L123(s_tilde[1], s_tilde[1], cn_353_HM1_batch, cn_353_HM2_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_353_HM1).to(device),torch.from_numpy(d_353_HM2).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask([s_tilde[1],s_tilde[1]], std_L2, cross=True)
        print('L2 prepared !')
        # L3
        print('Preparing L3...')
        wph_op.load_model(wph_model_cross)
        std_L3 = compute_std_L123(s_tilde[0], s_tilde[1], cn_217_FM_batch, cn_353_FM_batch)
        coeffs_L3 = wph_op.apply([torch.from_numpy(d_217_FM).to(device),torch.from_numpy(d_353_FM).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3),dim=0),torch.unsqueeze(torch.imag(coeffs_L3),dim=0)))
        mask_L3 = compute_mask([s_tilde[0],s_tilde[1]], std_L3, cross=True)
        print('L3 prepared !')
        # L4
        print('Preparing L4...')
        wph_op.load_model(wph_model_cross)
        std_L4 = compute_std_L45(s_tilde[0], cn_217_FM_batch)
        coeffs_L4 = wph_op.apply([torch.from_numpy(d_217_FM).to(device),torch.from_numpy(I).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_L4),dim=0),torch.unsqueeze(torch.imag(coeffs_L4),dim=0)))
        mask_L4 = compute_mask([s_tilde[0],torch.from_numpy(I).to(device)], std_L4, cross=True)
        print('L4 prepared !')
        # L5
        print('Preparing L5...')
        wph_op.load_model(wph_model_cross)
        std_L5 = compute_std_L45(s_tilde[1], cn_353_FM_batch)
        coeffs_L5 = wph_op.apply([torch.from_numpy(d_353_FM).to(device),torch.from_numpy(I).to(device)], norm=None, pbc=pbc, cross=True)
        coeffs_target_L5 = torch.cat((torch.unsqueeze(torch.real(coeffs_L5),dim=0),torch.unsqueeze(torch.imag(coeffs_L5),dim=0)))
        mask_L5 = compute_mask([s_tilde[1],torch.from_numpy(I).to(device)], std_L5, cross=True)
        print('L5 prepared !')
        # Minimization
        print('Beginning optimization...')
        result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params)
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, np.array([I,d_217_FM,s_tilde[0],d_217_FM-s_tilde[0],s_tilde0[0],d_353_FM,s_tilde[1],d_353_FM-s_tilde[1],s_tilde0[1]]))
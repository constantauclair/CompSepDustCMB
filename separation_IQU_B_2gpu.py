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
import multiprocessing as mp
from functools import partial

'''
This component separation algorithm aims to separate the polarized Q and U dust emission from the CMB 
and noise contamination on Planck-like data. The aim is to obtain the statistics of the Q and U dust 
emission as well the IQ and IU correlation. It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

The quantities involved are d (the total map), u (the optimized dust map), n (the noise + CMB map) for Q and U,
and I (the intensity map).

Loss terms:
    
# Dust QQ, UU and QU
L1 : (u_Q + n_Q_HM1) x (u_Q + n_Q_HM2) = d_Q_HM1 x d_Q_HM2                
L2 : (u_U + n_U_HM1) x (u_U + n_U_HM2) = d_U_HM1 x d_U_HM2                
L3 : (u_Q + n_Q_FM) x (u_U + n_U_FM) = d_Q_FM x d_U_FM 

# Dust IQ and IU
L4 : (u_Q + n_Q_FM) x I = d_Q_FM x I
L5 : (u_U + n_U_FM) x I = d_I_FM x I

# Noise + CMB QQ and UU
L6 : (d_Q_FM - u_Q) = n_Q_FM  
L7 : (d_U_FM - u_U) = n_U_FM  

''' 

###############################################################################
# INPUT PARAMETERS
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('freq', type=int) # 0 for 100, 1 for 143, 2 for 217 or 3 for 353
args = parser.parse_args()
freq = int(args.freq)
freqs = [100,143,217,353]

M, N = 768,768
J = 7
L = 4
method = 'L-BFGS-B'
pbc = False
dn = 5

file_name="separation_IQU_B_2gpu_"+str(freqs[freq])+"_3steps_30iters_.npy"

Mn = 50
n_step = 3
iter_per_step = 30

optim_params = {"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

devices = [0,1] # GPUs to use

batch_size = 5
n_batch = int(Mn/batch_size)

wph_model = ["S11","S00","S01","Cphase","C01","C00","L"]
wph_model_cross = ["S11","S00","S01","S10","Cphase","C01","C10","C00","L"]

###############################################################################
# DATA
###############################################################################

print('Starting IQU component separation at',freqs[freq],'GHz...')
print('Loading data...')

# Dust 
dust_maps = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_IQU_maps.npy').astype(np.float32)
d_Q_FM = dust_maps[1,freq,0]
d_Q_HM1 = dust_maps[1,freq,1]
d_Q_HM2 = dust_maps[1,freq,2]
d_U_FM = dust_maps[2,freq,0]
d_U_HM1 = dust_maps[2,freq,1]
d_U_HM2 = dust_maps[2,freq,2]
del dust_maps

print('Dust loaded !')

# CMB
cmb_maps = np.load('data/IQU_Planck_data/Sroll20_data/CMB_IQU_maps.npy').astype(np.float32)
c_Q = cmb_maps[1,freq,:Mn]
c_U = cmb_maps[2,freq,:Mn]
del cmb_maps

print('CMB loaded !')

# Noise
noise_Q = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_Noise_Q_'+str(freqs[freq])+'_maps.npy').astype(np.float32)
n_Q_FM = noise_Q[0,:Mn]
n_Q_HM1 = noise_Q[1,:Mn]
n_Q_HM2 = noise_Q[2,:Mn]
del noise_Q
noise_U = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_Noise_U_'+str(freqs[freq])+'_maps.npy').astype(np.float32)
n_U_FM = noise_U[0,:Mn]
n_U_HM1 = noise_U[1,:Mn]
n_U_HM2 = noise_U[2,:Mn]
del noise_U

print('Noise loaded !')

# I map
I = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_I_857_FM_768px.npy').astype(np.float32)

# CMB Noise combination
cn_Q_FM = c_Q + n_Q_FM
cn_Q_HM1 = c_Q + n_Q_HM1
cn_Q_HM2 = c_Q + n_Q_HM2

cn_U_FM = c_U + n_U_FM
cn_U_HM1 = c_U + n_U_HM1
cn_U_HM2 = c_U + n_U_HM2

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch

print('Creating batches...')

cn_Q_FM_batch = create_batch(torch.from_numpy(cn_Q_FM)).to(devices[0])
cn_Q_HM1_batch = create_batch(torch.from_numpy(cn_Q_HM1)).to(devices[0])
cn_Q_HM2_batch = create_batch(torch.from_numpy(cn_Q_HM2)).to(devices[0])

cn_U_FM_batch = create_batch(torch.from_numpy(cn_U_FM)).to(devices[0])
cn_U_HM1_batch = create_batch(torch.from_numpy(cn_U_HM1)).to(devices[0])
cn_U_HM2_batch = create_batch(torch.from_numpy(cn_U_HM2)).to(devices[0])

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
    return std.to(devices[0])

def compute_std_L45(u_A, conta_A):
    coeffs_ref = wph_op.apply([u_A,torch.from_numpy(I).to(devices[0])], norm=None, pbc=pbc, cross=True)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(n_batch):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure([u_A + conta_A[i],torch.from_numpy(I).expand(conta_A[i].size()).to(devices[0])], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return std.to(devices[0])

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
    return mean.to(devices[0]), std.to(devices[0])

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
    mask_real = torch.abs(torch.real(coeffs)).to(devices[0]) > thresh
    mask_imag = torch.abs(torch.imag(coeffs)).to(devices[0]) > thresh
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(devices[0])

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    thresh = get_thresh(coeffs)
    mask_real = torch.logical_and(torch.abs(torch.real(coeffs)).to(devices[0]) > thresh, std[0].to(devices[0]) > 0)
    mask_imag = torch.logical_and(torch.abs(torch.imag(coeffs)).to(devices[0]) > thresh, std[1].to(devices[0]) > 0)
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(devices[0])

def compute_L123(u_A,u_B,conta_A,conta_B,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(devices[0])
    std = std.to(devices[0])
    mask = mask.to(devices[0])
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([u_A + torch.from_numpy(conta_A[j]).to(devices[0]),u_B + torch.from_numpy(conta_B[j]).to(devices[0])], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L45(u_A,conta_A,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(devices[0])
    std = std.to(devices[0])
    mask = mask.to(devices[0])
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure([u_A + torch.from_numpy(conta_A[j]).to(devices[0]),torch.from_numpy(I).to(devices[0])], requires_grad=True, pbc=pbc, cross=True, mem_chunk_factor=50, mem_chunk_factor_grad=80)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc, cross=True)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_L67(u_A,obs_A,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(devices[0])
    std = std.to(devices[0])
    mask = mask.to(devices[0])
    loss_tot = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(obs_A).to(devices[0]) - u_A, requires_grad=True, pbc=pbc, cross=False, mem_chunk_factor=50, mem_chunk_factor_grad=80)
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

def objective_gpu_S11(u, objective_S11_stuff, wph_op, work_list, device_id):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    device = devices[device_id]
    wph_op.to(device)
    objective_S11_stuff = objective_S11_stuff.to(device)
    coeffs_target = objective_S11_stuff[0]
    std = objective_S11_stuff[1]
    mask = objective_S11_stuff[2]
    work_list = work_list[device_id]
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_Q = u[0]
    u_U = u[1]
    ######################
    L1 = compute_L123(u_Q,u_Q,cn_Q_HM1[work_list],cn_Q_HM2[work_list],coeffs_target[0],std[0],mask[0]) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_U,u_U,cn_U_HM1[work_list],cn_U_HM2[work_list],coeffs_target[1],std[1],mask[1]) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_Q,u_U,cn_Q_FM[work_list],cn_U_FM[work_list],coeffs_target[2],std[2],mask[2]) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    L4 = compute_L45(u_Q,cn_Q_FM[work_list],coeffs_target[3],std[3],mask[3]) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_U,cn_U_FM[work_list],coeffs_target[4],std[4],mask[4]) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    L6 = compute_L67(u_Q,d_Q_FM,coeffs_target[5],std[5],mask[5]) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_U,d_U_FM,coeffs_target[6],std[6],mask[6]) # Compute L7
    print("L7 = "+str(round(L7.item(),3)))
    ######################
    L = L1 + L2 + L3 + L4 + L5 + L6 + L7
    u_grad = u.grad.cpu().numpy().astype(u.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad

def objective_gpu(u, objective_stuff, wph_op, work_list, device_id):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    device = devices[device_id]
    wph_op.to(device)
    objective_stuff = objective_stuff.to(device)
    coeffs_target = objective_stuff[0]
    std = objective_stuff[1]
    mask = objective_stuff[2]
    work_list = work_list[device_id]
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_Q = u[0]
    u_U = u[1]
    ######################
    wph_op.load_model(wph_model_cross)
    L1 = compute_L123(u_Q,u_Q,cn_Q_HM1[work_list],cn_Q_HM2[work_list],coeffs_target[0],std[0],mask[0]) # Compute L1
    print("L1 = "+str(round(L1.item(),3)))
    L2 = compute_L123(u_U,u_U,cn_U_HM1[work_list],cn_U_HM2[work_list],coeffs_target[1],std[1],mask[1]) # Compute L2
    print("L2 = "+str(round(L2.item(),3)))
    L3 = compute_L123(u_Q,u_U,cn_Q_FM[work_list],cn_U_FM[work_list],coeffs_target[2],std[2],mask[2]) # Compute L3
    print("L3 = "+str(round(L3.item(),3)))
    ######################
    wph_op.load_model(wph_model_cross)
    L4 = compute_L45(u_Q,cn_Q_FM[work_list],coeffs_target[3],std[3],mask[3]) # Compute L4
    print("L4 = "+str(round(L4.item(),3)))
    L5 = compute_L45(u_U,cn_U_FM[work_list],coeffs_target[4],std[4],mask[4]) # Compute L5
    print("L5 = "+str(round(L5.item(),3)))
    ######################
    wph_op.load_model(wph_model)
    L6 = compute_L67(u_Q,d_Q_FM,coeffs_target[5],std[5],mask[5]) # Compute L6
    print("L6 = "+str(round(L6.item(),3)))
    L7 = compute_L67(u_U,d_U_FM,coeffs_target[6],std[6],mask[6]) # Compute L7
    print("L7 = "+str(round(L7.item(),3)))
    ######################
    L = L1 + L2 + L3 + L4 + L5 + L6 + L7
    u_grad = u.grad.cpu().numpy().astype(u.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad

def objective_S11(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((2, M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_gpu_S11, u, stuff_S11, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(2, M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L1 = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()

def objective(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((2, M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_gpu, u, stuff, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(2, M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L1 = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()

###############################################################################
# MINIMIZATION
###############################################################################

if __name__ == "__main__":
    total_start_time = time.time()
    print("Starting component separation...")
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=devices[0])
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization...")
    eval_cnt = 0
    s_tilde0 = np.array([d_Q_FM,d_U_FM])
    # L6
    print('Preparing L6...')
    coeffs_target_L6, std_L6 = compute_mean_std_L67(cn_Q_FM_batch)
    mask_L6 = compute_mask_S11(cn_Q_FM_batch[0,0])
    print('L6 prepared !')
    # L7
    print('Preparing L7...')
    coeffs_target_L7, std_L7 = compute_mean_std_L67(cn_U_FM_batch)
    mask_L7 = compute_mask_S11(cn_U_FM_batch[0,0])
    print('L7 prepared !')
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(devices[0]) # Initialization of the map
        # L1
        print('Preparing L1...')
        std_L1 = compute_std_L123(s_tilde0[0], s_tilde0[0], cn_Q_HM1_batch, cn_Q_HM2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_Q_HM1).to(devices[0]),torch.from_numpy(d_Q_HM2).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask_S11([s_tilde0[0],s_tilde0[0]], cross=True)
        print('L1 prepared !')
        # L2
        print('Preparing L2...')
        std_L2 = compute_std_L123(s_tilde0[1], s_tilde0[1], cn_U_HM1_batch, cn_U_HM2_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_U_HM1).to(devices[0]),torch.from_numpy(d_U_HM2).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask_S11([s_tilde0[1],s_tilde0[1]], cross=True)
        print('L2 prepared !')
        # L3
        print('Preparing L3..')
        std_L3 = compute_std_L123(s_tilde0[0], s_tilde0[1], cn_Q_FM_batch, cn_U_FM_batch)
        coeffs_L3 = wph_op.apply([torch.from_numpy(d_Q_FM).to(devices[0]),torch.from_numpy(d_U_FM).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3),dim=0),torch.unsqueeze(torch.imag(coeffs_L3),dim=0)))
        mask_L3 = compute_mask_S11([s_tilde0[0],s_tilde0[1]], cross=True)
        print('L3 prepared !')
        # L4
        print('Preparing L4...')
        std_L4 = compute_std_L45(s_tilde0[0], cn_Q_FM_batch)
        coeffs_L4 = wph_op.apply([torch.from_numpy(d_Q_FM).to(devices[0]),torch.from_numpy(I).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_L4),dim=0),torch.unsqueeze(torch.imag(coeffs_L4),dim=0)))
        mask_L4 = compute_mask_S11([s_tilde0[0],torch.from_numpy(I).to(devices[0])], cross=True)
        print('L4 prepared !')
        # L5
        print('Preparing L5...')
        std_L5 = compute_std_L45(s_tilde0[1], cn_U_FM_batch)
        coeffs_L5 = wph_op.apply([torch.from_numpy(d_U_FM).to(devices[0]),torch.from_numpy(I).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L5 = torch.cat((torch.unsqueeze(torch.real(coeffs_L5),dim=0),torch.unsqueeze(torch.imag(coeffs_L5),dim=0)))
        mask_L5 = compute_mask_S11([s_tilde0[1],torch.from_numpy(I).to(devices[0])], cross=True)
        print('L5 prepared !')
        # Gathering
        coeffs_target = [coeffs_target_L1,coeffs_target_L2,coeffs_target_L3,coeffs_target_L4,coeffs_target_L5,coeffs_target_L6,coeffs_target_L7]
        std = [std_L1,std_L2,std_L3,std_L4,std_L5,std_L6,std_L7]
        mask = [mask_L1,mask_L2,mask_L3,mask_L4,mask_L5,mask_L6,mask_L7]
        stuff_S11 = [coeffs_target,std,mask]
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
    coeffs_target_L6, std_L6 = compute_mean_std_L67(cn_Q_FM_batch)
    mask_L6 = compute_mask(cn_Q_FM_batch[0,0], std_L6)
    print('L6 prepared !')
    # L7
    print('Preparing L7...')
    wph_op.load_model(wph_model)
    coeffs_target_L7, std_L7 = compute_mean_std_L67(cn_U_FM_batch)
    mask_L7 = compute_mask(cn_U_FM_batch[0,0], std_L7)
    print('L7 prepared !')
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(devices[0]) # Initialization of the map
        # L1
        print('Preparing L1...')
        wph_op.load_model(wph_model_cross)
        std_L1 = compute_std_L123(s_tilde[0], s_tilde[0], cn_Q_HM1_batch, cn_Q_HM2_batch)
        coeffs_L1 = wph_op.apply([torch.from_numpy(d_Q_HM1).to(devices[0]),torch.from_numpy(d_Q_HM2).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_L1),dim=0),torch.unsqueeze(torch.imag(coeffs_L1),dim=0)))
        mask_L1 = compute_mask([s_tilde[0],s_tilde[0]], std_L1, cross=True)
        print('L1 prepared !')
        # L2
        print('Preparing L2...')
        wph_op.load_model(wph_model_cross)
        std_L2 = compute_std_L123(s_tilde[1], s_tilde[1], cn_U_HM1_batch, cn_U_HM2_batch)
        coeffs_L2 = wph_op.apply([torch.from_numpy(d_U_HM1).to(devices[0]),torch.from_numpy(d_U_HM2).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L2 = torch.cat((torch.unsqueeze(torch.real(coeffs_L2),dim=0),torch.unsqueeze(torch.imag(coeffs_L2),dim=0)))
        mask_L2 = compute_mask([s_tilde[1],s_tilde[1]], std_L2, cross=True)
        print('L2 prepared !')
        # L3
        print('Preparing L3...')
        wph_op.load_model(wph_model_cross)
        std_L3 = compute_std_L123(s_tilde[0], s_tilde[1], cn_Q_FM_batch, cn_U_FM_batch)
        coeffs_L3 = wph_op.apply([torch.from_numpy(d_Q_FM).to(devices[0]),torch.from_numpy(d_U_FM).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L3 = torch.cat((torch.unsqueeze(torch.real(coeffs_L3),dim=0),torch.unsqueeze(torch.imag(coeffs_L3),dim=0)))
        mask_L3 = compute_mask([s_tilde[0],s_tilde[1]], std_L3, cross=True)
        print('L3 prepared !')
        # L4
        print('Preparing L4...')
        wph_op.load_model(wph_model_cross)
        std_L4 = compute_std_L45(s_tilde[0], cn_Q_FM_batch)
        coeffs_L4 = wph_op.apply([torch.from_numpy(d_Q_FM).to(devices[0]),torch.from_numpy(I).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_L4),dim=0),torch.unsqueeze(torch.imag(coeffs_L4),dim=0)))
        mask_L4 = compute_mask([s_tilde[0],torch.from_numpy(I).to(devices[0])], std_L4, cross=True)
        print('L4 prepared !')
        # L5
        print('Preparing L5...')
        wph_op.load_model(wph_model_cross)
        std_L5 = compute_std_L45(s_tilde[1], cn_U_FM_batch)
        coeffs_L5 = wph_op.apply([torch.from_numpy(d_U_FM).to(devices[0]),torch.from_numpy(I).to(devices[0])], norm=None, pbc=pbc, cross=True)
        coeffs_target_L5 = torch.cat((torch.unsqueeze(torch.real(coeffs_L5),dim=0),torch.unsqueeze(torch.imag(coeffs_L5),dim=0)))
        mask_L5 = compute_mask([s_tilde[1],torch.from_numpy(I).to(devices[0])], std_L5, cross=True)
        print('L5 prepared !')
        # Gathering
        coeffs_target = [coeffs_target_L1,coeffs_target_L2,coeffs_target_L3,coeffs_target_L4,coeffs_target_L5,coeffs_target_L6,coeffs_target_L7]
        std = [std_L1,std_L2,std_L3,std_L4,std_L5,std_L6,std_L7]
        mask = [mask_L1,mask_L2,mask_L3,mask_L4,mask_L5,mask_L6,mask_L7]
        stuff = [coeffs_target,std,mask]
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
        np.save(file_name, np.array([I,d_Q_FM,s_tilde[0],d_Q_FM-s_tilde[0],s_tilde0[0],d_U_FM,s_tilde[1],d_U_FM-s_tilde[1],s_tilde0[1]]))
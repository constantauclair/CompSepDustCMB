# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import scipy.stats as stats
import argparse

'''
This component separation algorithm aims to separate the Stokes Q parameter of the polarized dust emission from the CMB 
and noise contamination on Planck-like mock data. This is done at 217 and 353 GHz. 
It makes use of the WPH statistics (see Régaldo-Saint Blancard et al. 2022). 
For any question: constant.auclair@phys.ens.fr
Another project has been led on the dust/CIB/noise separation on Herschel data (see Auclair et al. 2023).

The gradient descent is done on [u_353, u_D, u_CMB].

Loss terms:
    
L1 : u_353 + CMB + n_353 = d_353 (bias)
L2 : u_D + CMB_D + n_D = d_D (bias)
L3 : u_CMB = CMB (mean)
L4 : d_353 - u_353 - u_CMB = n_353 (mean)
L5 : d_D - u_D - CMB_D = n_D (mean)
L6 : (u_D + CMB_D + n_D) * T_353 = d_D * T_353 (bias)

'''
#######
# INPUT PARAMETERS
#######

M, N = 512, 512
J = 7
L = 4
dn = 5
pbc = True
method = 'L-BFGS-B'

parser = argparse.ArgumentParser()
parser.add_argument('channel', type=str)
parser.add_argument('loss_list', type=str)
parser.add_argument('fbm_slope', type=int)
args = parser.parse_args()
polar = args.channel
losses = args.loss_list
slope = args.fbm_slope

file_name="separation_multifreq_difference_Chameleon-Musca_"+polar+"_L"+losses+"_fbm"+str(slope)+".npy"

if polar == 'Q':
    polar_index = 1
if polar == 'U':
    polar_index = 2

n_step1 = 5
iter_per_step1 = 50
    
n_step2 = 10
iter_per_step2 = 100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 100 # Number of noises per iteration
batch_size = 10
n_batch = int(Mn/batch_size)

alpha = 0.17556374501554545

#######
# DATA
#######

Dust_217 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Dust_IQU_217.npy')[polar_index]
Dust_353 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Dust_IQU_353.npy')[polar_index]

T_Dust_353 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Dust_IQU_353.npy')[0]

CMB = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[polar_index,0]
    
CMB_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/CMB_IQU.npy')[polar_index]

Noise_217 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_217.npy')[polar_index,0]
Noise_353 = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_353.npy')[polar_index,0]
    
Noise_217_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_217.npy')[polar_index]
Noise_353_syn = np.load('data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_353.npy')[polar_index]

Dust_D = Dust_217 - alpha*Dust_353
CMB_D = (1-alpha)*CMB
CMB_D_syn = (1-alpha)*CMB_syn
Noise_D = Noise_217 - alpha*Noise_353
Noise_D_syn = Noise_217_syn - alpha*Noise_353_syn
    
Mixture_217 = Dust_217 + CMB + Noise_217
Mixture_353 = Dust_353 + CMB + Noise_353
Mixture_D = Mixture_217 - alpha*Mixture_353

print("SNR 353 =",np.std(Dust_353)/np.std(CMB+Noise_353))
print("SNR D =",np.std(Dust_D)/np.std(CMB_D+Noise_D))

#######
# USEFUL FUNCTIONS
#######

def power_spectrum(image):
    assert image.shape[0] == image.shape[1]    
    n = image.shape[0]
    fourier = np.fft.fftn(image)
    amplitude = (np.abs(fourier) ** 2).flatten()
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten() ** (1 / 2)
    kbins = np.arange(1 / 2, n // 2 + 1, 1)
    kvals = (kbins[1:] + kbins[:-1]) / 2
    bins, _, _ = stats.binned_statistic(knrm, amplitude, statistic = "mean", bins = kbins)
    return kvals, bins

def Gaussian(size, fwhm):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
def generate_fbm(noisy_data,noise,slope,frac=10):
    N = np.shape(noisy_data)[-1]
    k, noise_bins = power_spectrum(noise)
    _, noisy_data_bins = power_spectrum(noisy_data)
    estimated_data_bins = noisy_data_bins - noise_bins
    clean_mask = estimated_data_bins > noise_bins / frac
    k_crit = np.max(k[clean_mask])
    gauss = Gaussian(N,k_crit)
    noisy_data_FT = np.fft.fftshift(np.fft.fft2(noisy_data))
    filtered_noisy_data_FT = noisy_data_FT * gauss
    filtered_noisy_data = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_noisy_data_FT)))
    _, f_noisy_data_bins = power_spectrum(filtered_noisy_data)
    kfreq = np.fft.fftfreq(N) * N
    kfreq2D = np.meshgrid(kfreq, kfreq)
    random_phases = np.exp(1j*np.angle(np.fft.fft2(np.random.random(size=np.shape(noisy_data)))))
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2) ** (1 / 2)
    fbm_FT = knrm**(slope/2) * random_phases
    fbm_FT[0,0] = 1 * random_phases[0,0]
    k_fbm = int(k_crit/2)
    fbm_FT = fbm_FT / np.mean(np.abs(fbm_FT),where = knrm==k_fbm) * np.sqrt(f_noisy_data_bins[k_fbm])
    igauss = 1-Gaussian(N,k_fbm)
    f_fbm_FT = np.fft.ifftshift(np.fft.fftshift(fbm_FT)*igauss**4)
    fbm = np.real(np.fft.ifft2(f_fbm_FT))
    return filtered_noisy_data+fbm

def create_mono_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

Noise_353_batch = create_mono_batch(torch.from_numpy(Noise_353_syn).to(device), device=device)
Noise_D_batch = create_mono_batch(torch.from_numpy(Noise_D_syn).to(device), device=device)

CMB_batch = create_mono_batch(torch.from_numpy(CMB_syn).to(device), device=device)
CMB_D_batch = (1-alpha)*CMB_batch

def compute_coeffs_mean_std(mode,contamination_batch,x=None):
    coeffs_number = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).size(-1)
    ref_type = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).type()
    if mode in ['L1','L2']:
        COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
        coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(x.expand(contamination_batch[i].size()) + contamination_batch[i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].expand(coeffs_chunk.size()).type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
        std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    if mode in ['L3','L4','L5']:
        COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(contamination_batch[i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
        std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    if mode in ['L6']:
        COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
        coeffs_ref = wph_op.apply([x,torch.from_numpy(T_Dust_353).to(device)], norm=None, pbc=pbc, cross=True).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure([x.expand(contamination_batch[i].size()) + contamination_batch[i],torch.from_numpy(T_Dust_353).to(device).expand(contamination_batch[i].size())], pbc=pbc, cross=True)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc, cross=True)
                batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].expand(coeffs_chunk.size()).type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
        std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def compute_mask(x,std,cross=False):
    coeffs = wph_op.apply(x,norm=None,pbc=pbc,cross=cross)
    mask = torch.cat((torch.logical_and(torch.real(coeffs).to(device) > 1e-7, std[0].to(device) > 0),torch.logical_and(torch.imag(coeffs).to(device) > 1e-7, std[1].to(device) > 0)))
    return mask.to(device)

def compute_loss(x,coeffs_target,std,mask,cross=False):
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
    u = x.reshape((2, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_353 = u[0]
    u_D = u[1]
    loss_tot_1 = compute_loss(u_353,coeffs_target_L1,std_L1,mask_L1) # Compute the loss for 353
    loss_tot_2 = compute_loss(u_D,coeffs_target_L2,std_L2,mask_L2) # Compute the loss for D
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    loss_tot = loss_tot_1 + loss_tot_2
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L1 = "+str(round(loss_tot_1.item(),3)))
    print("L2 = "+str(round(loss_tot_2.item(),3)))
    print("")
    
    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((3, M, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    u_353 = u[0]
    u_D = u[1]
    u_CMB = u[2]
    L = 0 # Define total loss 
    # Compute the losses
    if '1' in losses:
        L1 = compute_loss('L1',u_353,coeffs_target_L1,std_L1,mask_L1)
        L = L + L1
    if '2' in losses:
        L2 = compute_loss('L2',u_D,coeffs_target_L2,std_L2,mask_L2)
        L = L + L2
    if '3' in losses:
        L3 = compute_loss('L3',u_CMB,coeffs_target_L3,std_L3,mask_L3)
        L = L + L3
    if '4' in losses:
        L4 = compute_loss('L4',torch.from_numpy(Mixture_353).to(device) - u_353 - u_CMB,coeffs_target_L4,std_L4,mask_L4)
        L = L + L4
    if '5' in losses:
        L5 = compute_loss('L5',torch.from_numpy(Mixture_D).to(device) - u_D - (1-alpha)*u_CMB,coeffs_target_L5,std_L5,mask_L5)
        L = L + L5
    if '6' in losses:
        L6 = compute_loss('L6',[u_D,torch.from_numpy(T_Dust_353).to(device)],coeffs_target_L6,std_L6,mask_L6,cross=True)
        L = L + L6
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Reshape the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    if '1' in losses:
        print("L1 = "+str(round(L1.item(),3)))
    if '2' in losses:
        print("L2 = "+str(round(L2.item(),3)))
    if '3' in losses:
        print("L3 = "+str(round(L3.item(),3)))
    if '4' in losses:
        print("L4 = "+str(round(L4.item(),3)))
    if '5' in losses:
        print("L5 = "+str(round(L5.item(),3)))
    if '6' in losses:
        print("L6 = "+str(round(L6.item(),3)))
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
    Initial_condition = np.array([generate_fbm(Mixture_353,Noise_353+CMB,slope),generate_fbm(Mixture_217,Noise_217+CMB,slope)-alpha*generate_fbm(Mixture_353,Noise_353+CMB,slope)])
    Dust_tilde0 = Initial_condition
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        print("Starting era "+str(i+1)+"...")
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        # Bias computation
        bias_L1, std_L1 = compute_coeffs_mean_std('L1',Noise_353_batch+CMB_batch,x=Dust_tilde0[0])
        bias_L2, std_L2 = compute_coeffs_mean_std('L2',Noise_D_batch+CMB_D_batch,x=Dust_tilde0[1])
        # Mask coputation
        mask_L1 = compute_mask(Dust_tilde0[0], std_L1).to(device)
        mask_L2 = compute_mask(Dust_tilde0[1], std_L2).to(device)
        # Coeffs target computation
        coeffs_L1 = wph_op.apply(torch.from_numpy(Mixture_353).to(device), norm=None, pbc=pbc)
        coeffs_target_L1 = torch.cat((torch.real(coeffs_L1) - bias_L1[0],torch.imag(coeffs_L1) - bias_L1[1]))
        coeffs_L2 = wph_op.apply(torch.from_numpy(Mixture_D).to(device), norm=None, pbc=pbc)
        coeffs_target_L2 = torch.cat((torch.real(coeffs_L2) - bias_L2[0],torch.imag(coeffs_L2) - bias_L2[1]))
        # Minimization
        result = opt.minimize(objective1, Dust_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((2, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")

    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    eval_cnt = 0
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    # Creating new set of variables
    Current_maps0 = np.array([Dust_tilde0[0],Dust_tilde0[1],np.random.normal(np.mean(CMB),np.std(CMB),size=(M,N))])
    # Computation of the coeffs target, std and mask
    if '3' in losses:
        coeffs_target_L3, std_L3 = compute_coeffs_mean_std('L3', CMB_batch)
        mask_L3 = compute_mask(torch.from_numpy(CMB).to(device),std_L3)
    if '4' in losses:
        coeffs_target_L4, std_L4 = compute_coeffs_mean_std('L4', Noise_353_batch)
        mask_L4 = compute_mask(torch.from_numpy(Noise_353).to(device),std_L4)
    if '5' in losses:
        coeffs_target_L5, std_L5 = compute_coeffs_mean_std('L5', Noise_D_batch)
        mask_L5 = compute_mask(torch.from_numpy(Noise_D).to(device),std_L5)
    Current_maps = Current_maps0
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        print("Starting era "+str(i+1)+"...")
        Current_maps = torch.from_numpy(Current_maps).to(device) # Initialization of the map
        # Computation of the coeffs target, std and mask
        if '1' in losses:
            bias_L1, std_L1 = compute_coeffs_mean_std('L1', Noise_353_batch+CMB_batch, x=Current_maps[0])
            coeffs_L1 = wph_op.apply(torch.from_numpy(Mixture_353).to(device), norm=None, pbc=pbc)
            coeffs_target_L1 = torch.cat((torch.real(coeffs_L1) - bias_L1[0],torch.imag(coeffs_L1) - bias_L1[1]))
            mask_L1 = compute_mask(Current_maps[0],std_L1)
        if '2' in losses:
            bias_L2, std_L2 = compute_coeffs_mean_std('L2', Noise_D_batch+CMB_D_batch, x=Current_maps[1])
            coeffs_L2 = wph_op.apply(torch.from_numpy(Mixture_D).to(device), norm=None, pbc=pbc)
            coeffs_target_L2 = torch.cat((torch.real(coeffs_L2) - bias_L2[0],torch.imag(coeffs_L2) - bias_L2[1]))
            mask_L2 = compute_mask(Current_maps[1],std_L2)
        if '6' in losses:
            bias_L6, std_L6 = compute_coeffs_mean_std('L6', Noise_D_batch+CMB_D_batch, x=Current_maps[1])
            coeffs_L6 = wph_op.apply(torch.from_numpy(Mixture_D).to(device), norm=None, pbc=pbc)
            coeffs_target_L6 = torch.cat((torch.real(coeffs_L6) - bias_L6[0],torch.imag(coeffs_L6) - bias_L6[1]))
            mask_L6 = compute_mask(Current_maps[1],std_L6,cross=True)
        # Minimization
        result = opt.minimize(objective2, Current_maps.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params2)
        final_loss, Current_maps, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        Current_maps = Current_maps.reshape((3, M, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, [T_Dust_353,Mixture_353,Dust_353,CMB,Noise_353,Current_maps[0],Current_maps[2],Mixture_353-Current_maps[0]-Current_maps[2],Mixture_D,Dust_D,CMB_D,Noise_D,Current_maps[1],(1-alpha)*Current_maps[2],Mixture_D-Current_maps[1]-(1-alpha)*Current_maps[2]])        
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw

#######
# INPUT PARAMETERS
#######

n_freq = 2

M, N = 256, 256
J = 6
L = 4
dn = 2
pbc = True

SNR = 1

file_name="separation_multifreq_L123_oldstyle.npy"

n_step1 = 1
iter_per_step1 = 200

n_step2 = 10
iter_per_step2 = 100

optim_params1 = {"maxiter": iter_per_step1, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": iter_per_step2, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

device = 0 # GPU to use

Mn = 20 # Number of noises per iteration

#######
# DATA
#######

Dust_1 = np.load('data/realistic_data/Dust_EE_217_microK.npy')
Dust_2 = np.load('data/realistic_data/Dust_EE_353_microK.npy')

CMB = np.load('data/realistic_data/CMB_EE_8arcmin_microK.npy')[0]

CMB_syn = np.load('data/realistic_data/CMB_EE_8arcmin_microK.npy')[1:Mn+1]

Noise_1 = np.load('data/realistic_data/Noise_EE_217_8arcmin_microK.npy')[0]
Noise_2 = np.load('data/realistic_data/Noise_EE_353_8arcmin_microK.npy')[0]

Noise_1_syn = np.load('data/realistic_data/Noise_EE_217_8arcmin_microK.npy')[1:Mn+1]
Noise_2_syn = np.load('data/realistic_data/Noise_EE_353_8arcmin_microK.npy')[1:Mn+1]

Mixture_1 = Dust_1 + CMB + Noise_1
Mixture_2 = Dust_2 + CMB + Noise_2

print("SNR F1 =",np.std(Dust_1)/np.std(CMB+Noise_1))
print("SNR F2 =",np.std(Dust_2)/np.std(CMB+Noise_2))

## Define final variables

Mixture = np.array([Mixture_1,Mixture_2])

Dust = np.array([Dust_1,Dust_2])

CMB_Noise = np.array([CMB+Noise_1,CMB+Noise_2])

CMB_Noise_syn = np.array([CMB_syn+Noise_1_syn,CMB_syn+Noise_2_syn])

#######
# USEFUL FUNCTIONS
#######

def compute_std_L1(x):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    for freq in range(n_freq):
        u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + torch.from_numpy(CMB_Noise_syn[freq]).to(device), pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
            COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
            del coeffs_chunk, indices
        del u_noisy, nb_chunks
    mean = torch.mean(COEFFS,axis=1)
    std = torch.std(COEFFS,axis=1)
    return mean, std

def compute_complex_mean_std_L1(x):
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    (_,coeffs_number) = np.shape(coeffs_ref)
    COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=coeffs_ref.type())
    for freq in range(n_freq):
        u_noisy, nb_chunks = wph_op.preconfigure(x[freq] + torch.from_numpy(CMB_Noise_syn[freq]).to(device), pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
            COEFFS[freq,:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
            del coeffs_chunk, indices
        del u_noisy, nb_chunks
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
    return mean, std

def compute_complex_mean_std_L3(x):
    coeffs_ref = wph_op.apply([x[0],x[1]], norm=None, cross=True, pbc=pbc)
    coeffs_number = len(coeffs_ref)
    pairs = []
    for i in range(Mn):
        for j in range(Mn):
            if j>i:
                pairs.append([x[0].cpu()+torch.from_numpy(CMB_Noise_syn[0,i]),x[1].cpu()+torch.from_numpy(CMB_Noise_syn[1,j])])
    n_pairs = len(pairs)
    pairs = torch.from_numpy(np.array(pairs)).to(device)
    COEFFS = torch.zeros((n_pairs,coeffs_number)).type(dtype=coeffs_ref.type())
    computed_pairs = 0
    for i in range(int(np.ceil(n_pairs/Mn))):
        if n_pairs - computed_pairs > Mn:
            uu_noisy, nb_chunks = wph_op.preconfigure([pairs[computed_pairs:computed_pairs+Mn,0],pairs[computed_pairs:computed_pairs+Mn,1]], cross=True, pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(uu_noisy, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                COEFFS[computed_pairs:computed_pairs+Mn,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            computed_pairs += Mn
            del uu_noisy, nb_chunks
        if n_pairs - computed_pairs <= Mn:
            uu_noisy, nb_chunks = wph_op.preconfigure([pairs[computed_pairs:,0],pairs[computed_pairs:,1]], cross=True, pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(uu_noisy, j, norm=None, cross=True, ret_indices=True, pbc=pbc)
                COEFFS[computed_pairs:,indices] = coeffs_chunk.type(dtype=coeffs_ref.type())
                del coeffs_chunk, indices
            del uu_noisy, nb_chunks
    mean = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return mean, std

#######
# OBJECTIVE FUNCTIONS
#######

def objective1(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_freq, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss
    loss_tot_F1 = torch.zeros(1)
    loss_tot_F2 = torch.zeros(1)
    for i in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(u + torch.from_numpy(CMB_Noise_syn[:,i]).to(device), requires_grad=True, pbc=pbc)
        for j in range(nb_chunks):
            if pbc==True:
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                loss_F1 = torch.sum(torch.abs( (coeffs_chunk[0] - coeffs_target[0,indices]) / std[0,indices] ) ** 2)
                loss_F2 = torch.sum(torch.abs( (coeffs_chunk[1] - coeffs_target[1,indices]) / std[1,indices] ) ** 2)
                loss_F1 = loss_F1 / len(coeffs_target[0])
                loss_F2 = loss_F2 / len(coeffs_target[1])
                loss_F1.backward(retain_graph=True)
                loss_F2.backward(retain_graph=True)
                loss_tot_F1 += loss_F1.detach().cpu()
                loss_tot_F2 += loss_F2.detach().cpu()
            if pbc==False:
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                kept_coeffs_F1 = torch.nan_to_num(relevant_coeffs_step1_L1_F1[indices] / std[0,indices],nan=0)
                kept_coeffs_F2 = torch.nan_to_num(relevant_coeffs_step1_L1_F2[indices] / std[1,indices],nan=0)
                loss_F1 = torch.sum(torch.abs( (coeffs_chunk[0] - coeffs_target[0,indices]) * kept_coeffs_F1 ) ** 2)
                loss_F2 = torch.sum(torch.abs( (coeffs_chunk[1] - coeffs_target[1,indices]) * kept_coeffs_F2 ) ** 2)
                loss_F1 = loss_F1 / coeffs_number_step1_L1_F1
                loss_F2 = loss_F2 / coeffs_number_step1_L1_F2
                loss_F1.backward(retain_graph=True)
                loss_F2.backward(retain_graph=True)
                loss_tot_F1 += loss_F1.detach().cpu()
                loss_tot_F2 += loss_F2.detach().cpu()
            del coeffs_chunk, indices, loss_F1, loss_F2
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_F1 + loss_tot_F2
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L F1 = "+str(round(loss_tot_F1.item(),3)))
    print("L F2 = "+str(round(loss_tot_F2.item(),3)))
    print("")
    
    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

def objective2(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    u = x.reshape((n_freq, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Compute the loss 1
    loss_tot_1_F1_real = torch.zeros(1)
    loss_tot_1_F2_real = torch.zeros(1)
    loss_tot_1_F1_imag = torch.zeros(1)
    loss_tot_1_F2_imag = torch.zeros(1)
    for j in range(Mn):
        u_noisy, nb_chunks = wph_op.preconfigure(u + torch.from_numpy(CMB_Noise_syn[:,j]).to(device), requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, i, norm=None, ret_indices=True, pbc=pbc)
            # Loss F1
            loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L1[0,0][indices]) / std_L1[0,0][indices] ) ** 2)
            kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std_L1[1,0][indices],nan=0)
            loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L1[1,0][indices]) * kept_coeffs ) ** 2)
            loss_F1_real = loss_F1_real / len(indices) #real_coeffs_number_L1
            loss_F1_imag = loss_F1_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L1
            # Loss F2
            loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L1[0,1][indices]) / std_L1[0,1][indices] ) ** 2)
            kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L1[indices] / std_L1[1,1][indices],nan=0)
            loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L1[1,1][indices]) * kept_coeffs ) ** 2)
            loss_F2_real = loss_F2_real / len(indices) #real_coeffs_number_L1
            loss_F2_imag = loss_F2_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L1
            #
            loss_F1_real.backward(retain_graph=True)
            loss_F1_imag.backward(retain_graph=True)
            loss_F2_real.backward(retain_graph=True)
            loss_F2_imag.backward(retain_graph=True)
            #
            loss_tot_1_F1_real += loss_F1_real.detach().cpu()
            loss_tot_1_F1_imag += loss_F1_imag.detach().cpu()
            loss_tot_1_F2_real += loss_F2_real.detach().cpu()
            loss_tot_1_F2_imag += loss_F2_imag.detach().cpu()
            del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 2
    loss_tot_2_F1_real = torch.zeros(1)
    loss_tot_2_F2_real = torch.zeros(1)
    loss_tot_2_F1_imag = torch.zeros(1)
    loss_tot_2_F2_imag = torch.zeros(1)
    u_bis, nb_chunks = wph_op.preconfigure(torch.from_numpy(Mixture).to(device) - u, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_bis, i, norm=None, ret_indices=True, pbc=pbc)
        # Loss F1
        loss_F1_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[0]) - coeffs_target_L2[0,0][indices]) / std_L2[0,0][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_L2[1,0][indices],nan=0)
        loss_F1_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[0]) - coeffs_target_L2[1,0][indices]) * kept_coeffs ) ** 2)
        loss_F1_real = loss_F1_real / real_coeffs_number_L2
        loss_F1_imag = loss_F1_imag / imag_coeffs_number_L2
        # Loss F2
        loss_F2_real = torch.sum(torch.abs( (torch.real(coeffs_chunk[1]) - coeffs_target_L2[0,1][indices]) / std_L2[0,1][indices] ) ** 2)
        kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L2[indices] / std_L2[1,1][indices],nan=0)
        loss_F2_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk[1]) - coeffs_target_L2[1,1][indices]) * kept_coeffs ) ** 2)
        loss_F2_real = loss_F2_real / real_coeffs_number_L2
        loss_F2_imag = loss_F2_imag / imag_coeffs_number_L2
        #
        loss_F1_real.backward(retain_graph=True)
        loss_F1_imag.backward(retain_graph=True)
        loss_F2_real.backward(retain_graph=True)
        loss_F2_imag.backward(retain_graph=True)
        #
        loss_tot_2_F1_real += loss_F1_real.detach().cpu()
        loss_tot_2_F1_imag += loss_F1_imag.detach().cpu()
        loss_tot_2_F2_real += loss_F2_real.detach().cpu()
        loss_tot_2_F2_imag += loss_F2_imag.detach().cpu()
        del coeffs_chunk, indices, loss_F1_real, loss_F1_imag, loss_F2_real, loss_F2_imag
        
    # Compute the loss 3
    loss_tot_3_real = torch.zeros(1)
    loss_tot_3_imag = torch.zeros(1)
    for j in range(Mn-1):
        u_ter, nb_chunks = wph_op.preconfigure([u[0] + torch.from_numpy(CMB_Noise_syn[0,j]).to(device),u[1] + torch.from_numpy(CMB_Noise_syn[0,j+1]).to(device)], cross=True, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_ter, i, norm=None, cross=True, ret_indices=True, pbc=pbc)
            #
            loss_real = torch.sum(torch.abs( (torch.real(coeffs_chunk) - coeffs_target_L3[0][indices]) / std_L3[0][indices] ) ** 2)
            kept_coeffs = torch.nan_to_num(relevant_imaginary_coeffs_L3[indices] / std_L3[1][indices],nan=0)
            loss_imag = torch.sum(torch.abs( (torch.imag(coeffs_chunk) - coeffs_target_L3[1][indices]) * kept_coeffs ) ** 2)
            loss_real = loss_real / len(indices) #real_coeffs_number_L3
            loss_imag = loss_imag / torch.where(torch.sum(torch.where(kept_coeffs>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs>0,1,0))) #imag_coeffs_number_L3
            #
            loss_real.backward(retain_graph=True)
            loss_imag.backward(retain_graph=True)
            #
            loss_tot_3_real += loss_real.detach().cpu()
            loss_tot_3_imag += loss_imag.detach().cpu()
            del coeffs_chunk, indices, loss_real, loss_imag
    
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    loss_tot = loss_tot_1_F1_real + loss_tot_1_F1_imag + loss_tot_1_F2_real + loss_tot_1_F2_imag + loss_tot_2_F1_real + loss_tot_2_F1_imag + loss_tot_2_F2_real + loss_tot_2_F2_imag + loss_tot_3_real + loss_tot_3_imag
    
    print("L = "+str(round(loss_tot.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("L1 F1 real = "+str(round(loss_tot_1_F1_real.item(),3)))
    print("L1 F1 imag = "+str(round(loss_tot_1_F1_imag.item(),3)))
    print("L1 F2 real = "+str(round(loss_tot_1_F2_real.item(),3)))
    print("L1 F2 imag = "+str(round(loss_tot_1_F2_imag.item(),3)))
    print("L2 F1 real = "+str(round(loss_tot_2_F1_real.item(),3)))
    print("L2 F1 imag = "+str(round(loss_tot_2_F1_imag.item(),3)))
    print("L2 F2 real = "+str(round(loss_tot_2_F2_real.item(),3)))
    print("L2 F2 imag = "+str(round(loss_tot_2_F2_imag.item(),3)))
    print("L3 real = "+str(round(loss_tot_3_real.item(),3)))
    print("L3 imag = "+str(round(loss_tot_3_imag.item(),3)))
    print("")

    eval_cnt += 1
    return loss_tot.item(), u_grad.ravel()

#######
# MINIMIZATION
#######

if __name__ == "__main__":
    
    total_start_time = time.time()
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    ## First minimization
    print("Starting first minimization (only S11)...")
    
    eval_cnt = 0
    
    Dust_tilde0 = np.array([Mixture_1,Mixture_2])
    
    if pbc==False:
        # Identification of the irrelevant imaginary parts of the coeffs
        # F1
        coeffs_step1_L1_F1 = torch.abs(wph_op.apply(torch.from_numpy(Dust_tilde0[0]).to(device),norm=None,pbc=pbc))
        relevant_coeffs_step1_L1_F1 = torch.where(coeffs_step1_L1_F1 > 1e-6,1,0)
        # F2
        coeffs_step1_L1_F2 = torch.abs(wph_op.apply(torch.from_numpy(Dust_tilde0[1]).to(device),norm=None,pbc=pbc))
        relevant_coeffs_step1_L1_F2 = torch.where(coeffs_step1_L1_F2 > 1e-6,1,0)
        
        # Computation of the std
        std = compute_std_L1(torch.from_numpy(Dust_tilde0).to(device))
        
        # Compute the number of coeffs
        # F1
        kept_coeffs_step1_L1_F1 = torch.nan_to_num(relevant_coeffs_step1_L1_F1 / std[0],nan=0)
        coeffs_number_step1_L1_F1 = torch.where(torch.sum(torch.where(kept_coeffs_step1_L1_F1>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_step1_L1_F1>0,1,0))).item()
        # F2
        kept_coeffs_step1_L1_F2 = torch.nan_to_num(relevant_coeffs_step1_L1_F2 / std[1],nan=0)
        coeffs_number_step1_L1_F2 = torch.where(torch.sum(torch.where(kept_coeffs_step1_L1_F2>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_step1_L1_F2>0,1,0))).item()
        
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        _, std = compute_std_L1(torch.from_numpy(Mixture).to(device))
        
        # Coeffs target computation
        coeffs_target = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        
        # Minimization
        result = opt.minimize(objective1, Dust_tilde0.cpu().ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((n_freq, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
           
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Identification of the irrelevant imaginary parts of the coeffs
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    coeffs_imag_L1 = torch.imag(wph_op.apply(Dust_tilde0[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L1 = torch.where(torch.abs(coeffs_imag_L1) > 1e-6,1,0)
    coeffs_imag_L2 = torch.imag(wph_op.apply(CMB_Noise[0],norm=None,pbc=pbc))
    relevant_imaginary_coeffs_L2 = torch.where(torch.abs(coeffs_imag_L2) > 1e-6,1,0)
    coeffs_imag_L3 = torch.imag(wph_op.apply([Dust_tilde0[0],Dust_tilde0[1]],norm=None,cross=True,pbc=pbc))
    relevant_imaginary_coeffs_L3 = torch.where(torch.abs(coeffs_imag_L3) > 1e-6,1,0)
    
    # Computation of the coeffs and std
    _, std_L1 = compute_complex_mean_std_L1(torch.from_numpy(Dust).to(device))
    mean_L2, std_L2 = compute_complex_mean_std_L1(torch.from_numpy(Dust*0).to(device))
    _, std_L3 = compute_complex_mean_std_L3(torch.from_numpy(Dust).to(device))
    
    # Compute the number of coeffs
    # L1
    real_coeffs_number_L1 = len(torch.real(wph_op.apply(torch.from_numpy(Dust_tilde0[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_L1 = torch.nan_to_num(relevant_imaginary_coeffs_L1 / std[1,0],nan=0)
    imag_coeffs_number_L1 = torch.where(torch.sum(torch.where(kept_coeffs_L1>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L1>0,1,0))).item()
    # L2
    real_coeffs_number_L2 = len(torch.real(wph_op.apply(torch.from_numpy(CMB_Noise[0]).to(device),norm=None,pbc=pbc)))
    kept_coeffs_L2 = torch.nan_to_num(relevant_imaginary_coeffs_L2 / std_L2[1,0],nan=0)
    imag_coeffs_number_L2 = torch.where(torch.sum(torch.where(kept_coeffs_L2>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L2>0,1,0))).item()
    # L3
    real_coeffs_number_L3 = len(torch.real(wph_op.apply([torch.from_numpy(Dust_tilde0[0]).to(device),torch.from_numpy(Dust_tilde0[1]).to(device)],norm=None,cross=True,pbc=pbc)))
    kept_coeffs_L3 = torch.nan_to_num(relevant_imaginary_coeffs_L3 / std_L3[1],nan=0)
    imag_coeffs_number_L3 = torch.where(torch.sum(torch.where(kept_coeffs_L3>0,1,0))==0,1,torch.sum(torch.where(kept_coeffs_L3>0,1,0))).item()
    
    Dust_tilde = Dust_tilde0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Dust_tilde = torch.from_numpy(Dust_tilde).to(device)
        
        # Bias computation
        _, std_L1 = compute_complex_mean_std_L1(Dust)
        _, std_L3 = compute_complex_mean_std_L3(Dust)
        
        # Coeffs target computation
        # L1
        coeffs_target_L1 = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        # L2
        coeffs_target_L2 = mean_L2
        # L3
        coeffs_target_L3 = wph_op.apply([torch.from_numpy(Mixture[0]),torch.from_numpy(Mixture[1])], norm=None, cross=True, pbc=pbc)
        
        # Minimization
        result = opt.minimize(objective2, torch.from_numpy(Dust_tilde0).ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
        final_loss, Dust_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde = Dust_tilde.reshape((n_freq, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,CMB_Noise,Dust_tilde,Mixture-Dust_tilde,Dust_tilde0,Mixture-Dust_tilde0])        
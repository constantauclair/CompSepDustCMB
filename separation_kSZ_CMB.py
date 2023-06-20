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
args = parser.parse_args()
threshold = args.thresh

fac = 100

M, N = 500, 500
J = 7
L = 4
dn = 5
pbc = True
method = 'L-BFGS-B'

file_name="separation_kSZ_CMB_fac="+str(fac)+"_threshold="+str(int(100*threshold))+".npy"

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

#######
# DATA
#######

kSZ = np.load('data/kSZ_CMB_data/kSZ.npy') * fac

CMB = np.load('data/kSZ_CMB_data/CMB.npy')

CMB_syn = np.load('data/kSZ_CMB_data/CMB_syn.npy')

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
    
def compute_mask(st_calc,x,threshold,norm=True):
    return scat.compute_threshold_mask(st_calc.scattering_cov_constant(x,normalization=norm),threshold)
    
def compute_coeffs(st_calc,x,mask,norm=False,use_ref=False):
    coeffs = scat.threshold_coeffs(st_calc.scattering_cov_constant(x,normalization=norm,use_ref=use_ref),mask)
    if len(np.shape(x)) == 2:
        return coeffs[0]
    if len(np.shape(x)) == 3:
        return coeffs
    
def create_mono_batch(n, device):
    batch = torch.zeros([n_batch,batch_size,M,N])
    for i in range(n_batch):
        batch[i] = n[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

CMB_batch = create_mono_batch(torch.from_numpy(CMB_syn).to(device), device=device)

def compute_coeffs_mean_std_L1(x,only_S11=False):
    coeffs_number = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).size(-1)
    ref_type = wph_op.apply(contamination_batch[0,0], norm=None, pbc=pbc).type()
    # Mode for L1 
    if mode == 'classic_bias':
        COEFFS = torch.zeros((n_freq,Mn,coeffs_number)).type(dtype=ref_type)
        coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc).type(dtype=ref_type)
        computed_conta = 0
        for i in range(n_batch):
            batch_COEFFS = torch.zeros((n_freq,batch_size,coeffs_number)).type(dtype=ref_type)
            u_noisy, nb_chunks = wph_op.preconfigure(x.unsqueeze(1).expand(contamination_batch[:,i].size()) + contamination_batch[:,i], pbc=pbc)
            for j in range(nb_chunks):
                coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=None, ret_indices=True, pbc=pbc)
                batch_COEFFS[:,:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[:,indices].unsqueeze(1).expand(coeffs_chunk.size()).type(dtype=ref_type)
                del coeffs_chunk, indices
            COEFFS[:,computed_conta:computed_conta+batch_size] = batch_COEFFS
            computed_conta += batch_size
            del u_noisy, nb_chunks, batch_COEFFS
            sys.stdout.flush() # Flush the standard output
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=1),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=1),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=1),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=1)
            std = torch.std(COEFFS,axis=1)
    # Mode for L2
    if mode == 'mean_monofreq':
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
        if real_imag:
            bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
            std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
        else:
            bias = torch.mean(COEFFS,axis=0)
            std = torch.std(COEFFS,axis=0)

def compute_loss(mode,x,coeffs_target,std,mask):
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    # Mode for the first iteration step
    if mode == 'first_iter':
        loss_tot_F1 = torch.zeros(1)
        loss_tot_F2 = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
            # Loss F1
            loss_F1 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[0])[mask[0,0,indices]] - coeffs_target[0,0][indices][mask[0,0,indices]]) / std[0,0][indices][mask[0,0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[0])[mask[1,0,indices]] - coeffs_target[1,0][indices][mask[1,0,indices]]) / std[1,0][indices][mask[1,0,indices]] ) ** 2) ) / ( mask[0,0].sum() + mask[1,0].sum() )
            loss_F1.backward(retain_graph=True)
            loss_tot_F1 += loss_F1.detach().cpu()
            # Loss F2
            loss_F2 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[1])[mask[0,1,indices]] - coeffs_target[0,1][indices][mask[0,1,indices]]) / std[0,1][indices][mask[0,1,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[1])[mask[1,1,indices]] - coeffs_target[1,1][indices][mask[1,1,indices]]) / std[1,1][indices][mask[1,1,indices]] ) ** 2) ) / ( mask[0,1].sum() + mask[1,1].sum() )
            loss_F2.backward(retain_graph=True)
            loss_tot_F2 += loss_F2.detach().cpu()
            #
            del coeffs_chunk, indices, loss_F1, loss_F2
        return loss_tot_F1, loss_tot_F2
    # Mode for L1 and L3
    if mode in ['L1','L3']:
        loss_tot_F1 = torch.zeros(1)
        loss_tot_F2 = torch.zeros(1)
        u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
            # Loss F1
            loss_F1 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[0])[mask[0,0,indices]] - coeffs_target[0,0][indices][mask[0,0,indices]]) / std[0,0][indices][mask[0,0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[0])[mask[1,0,indices]] - coeffs_target[1,0][indices][mask[1,0,indices]]) / std[1,0][indices][mask[1,0,indices]] ) ** 2) ) / ( mask[0,0].sum() + mask[1,0].sum() )
            loss_F1.backward(retain_graph=True)
            loss_tot_F1 += loss_F1.detach().cpu()
            # Loss F2
            loss_F2 = ( torch.sum(torch.abs( (torch.real(coeffs_chunk[1])[mask[0,1,indices]] - coeffs_target[0,1][indices][mask[0,1,indices]]) / std[0,1][indices][mask[0,1,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk[1])[mask[1,1,indices]] - coeffs_target[1,1][indices][mask[1,1,indices]]) / std[1,1][indices][mask[1,1,indices]] ) ** 2) ) / ( mask[0,1].sum() + mask[1,1].sum() )
            loss_F2.backward(retain_graph=True)
            loss_tot_F2 += loss_F2.detach().cpu()
            #
            del coeffs_chunk, indices, loss_F1, loss_F2
        return loss_tot_F1, loss_tot_F2

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
    loss_tot_F1, loss_tot_F2 = compute_loss('first_iter',u,coeffs_target,std,mask)
    
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
    u = x.reshape((n_maps, M, N))
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    u_dust = u[:n_freq]
    u_CMB = u[n_freq]
    
    # Define total loss 
    L = 0
    
    # Compute the losses
    if '1' in losses:
        L1_F1, L1_F2 = compute_loss('L1',u_dust,coeffs_target_L1,std_L1,mask_L1)
        L = L + L1_F1 + L1_F2
    if '2' in losses:
        L2 = compute_loss('L2',u_CMB,coeffs_target_L2,std_L2,mask_L2)
        L = L + L2
    if '3' in losses:
        L3_F1, L3_F2 = compute_loss('L3',torch.from_numpy(Mixture).to(device) - u_dust - u_CMB,coeffs_target_L3,std_L3,mask_L3)
        L = L + L3_F1 + L3_F2
    if '4' in losses:
        L4 = compute_loss('L4',[u_dust[0],u_dust[1]],coeffs_target_L4,std_L4,mask_L4)
        L = L + L4
    if '5' in losses:
        L5 = compute_loss('L5',[torch.from_numpy(Mixture[0]).to(device) - u_dust[0] - u_CMB,torch.from_numpy(Mixture[1]).to(device) - u_dust[1] - u_CMB],coeffs_target_L5,std_L5,mask_L5)
        L = L + L5
    if '6' in losses:
        L6_F1, L6_F2 = compute_loss('L6',[u_dust,u_CMB.expand((n_freq,M,N))],coeffs_target_L6,std_L6,mask_L6)
        L = L + L6_F1 + L6_F2
    if '7' in losses:
        L7_F1, L7_F2 = compute_loss('L7',[u_dust,torch.from_numpy(Mixture).to(device) - u_dust - u_CMB],coeffs_target_L7,std_L7,mask_L7)
        L = L + L6_F1 + L6_F2
    if '8' in losses:
        L8_F1, L8_F2 = compute_loss('L8',[u_CMB.expand((n_freq,M,N)),torch.from_numpy(Mixture).to(device) - u_dust - u_CMB],coeffs_target_L8,std_L8,mask_L8)
        L = L + L7_F1 + L7_F2
        
    # Reshape the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)
    
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    if '1' in losses:
        print("L1 F1 = "+str(round(L1_F1.item(),3)))
        print("L1 F2 = "+str(round(L1_F2.item(),3)))
    if '2' in losses:
        print("L2 = "+str(round(L2.item(),3)))
    if '3' in losses:
        print("L3 F1 = "+str(round(L3_F1.item(),3)))
        print("L3 F2 = "+str(round(L3_F2.item(),3)))
    if '4' in losses:
        print("L4 = "+str(round(L4.item(),3)))
    if '5' in losses:
        print("L5 = "+str(round(L5.item(),3)))
    if '6' in losses:
        print("L6 F1 = "+str(round(L6_F1.item(),3)))
        print("L6 F2 = "+str(round(L6_F2.item(),3)))
    if '7' in losses:
        print("L7 F1 = "+str(round(L7_F1.item(),3)))
        print("L7 F2 = "+str(round(L7_F2.item(),3)))
    if '8' in losses:
        print("L8 F1 = "+str(round(L8_F1.item(),3)))
        print("L8 F2 = "+str(round(L8_F2.item(),3)))
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
    
    Initial_condition = np.array([generate_fbm(Mixture_1,Noise_1+CMB,slope),generate_fbm(Mixture_2,Noise_2+CMB,slope)])
    
    Dust_tilde0 = Initial_condition
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step1):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Dust_tilde0 = torch.from_numpy(Dust_tilde0).to(device)
        
        # Bias computation
        bias, std = compute_coeffs_mean_std('classic_bias',Noise_batch+CMB_batch.expand(Noise_batch.size()),x=Dust_tilde0)
        
        # Mask coputation
        mask = compute_mask(Dust_tilde0, std).to(device)
        
        # Coeffs target computation
        coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias[1],dim=0)))
        
        # Minimization
        result = opt.minimize(objective1, Dust_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params1)
        final_loss, Dust_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Dust_tilde0 = Dust_tilde0.reshape((n_freq, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second step of minimization (all coeffs)...")
    
    eval_cnt = 0
    
    # Initializing operator
    wph_op.load_model(["S11","S00","S01","Cphase","C01","C00","L"])
    wph_op.clear_normalization()
    
    # Creating new set of variables
    Current_maps0 = np.array([Dust_tilde0[0],Dust_tilde0[1],np.random.normal(np.mean(CMB),np.std(CMB),size=(M,N))])
    
    # Computation of the coeffs and std
    if '2' in losses:
        coeffs_target_L2, std_L2 = compute_coeffs_mean_std('mean_monofreq', CMB_batch)
    if '3' in losses:
        coeffs_target_L3, std_L3 = compute_coeffs_mean_std('mean', Noise_batch)
    if '5' in losses:
        coeffs_target_L5, std_L5 = compute_coeffs_mean_std('cross_freq_mean', Noise_batch)
    if '8' in losses:
        coeffs_target_L8, std_L8 = compute_coeffs_mean_std('cross_mean', CMB_batch.expand(Noise_batch.size()), cross_contamination_batch=Noise_batch)
    
    # Mask computation
    if '2' in losses:
        mask_L2 = compute_mask(torch.from_numpy(CMB).to(device),std_L2)
    if '3' in losses:
        mask_L3 = compute_mask(torch.from_numpy(Noise).to(device),std_L3)
    if '5' in losses:
        mask_L5 = compute_mask([torch.from_numpy(Noise[0]).to(device),torch.from_numpy(Noise[1]).to(device)],std_L5,cross=True)
    if '8' in losses:
        mask_L8 = compute_mask([torch.from_numpy(CMB).to(device).expand((n_freq,M,N)),torch.from_numpy(Noise).to(device)],std_L8,cross=True)
    
    Current_maps = Current_maps0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    for i in range(n_step2):
        
        print("Starting era "+str(i+1)+"...")
        
        # Initialization of the map
        Current_maps = torch.from_numpy(Current_maps).to(device)
        
        # Bias computation
        if '1' in losses:
            bias_L1, std_L1 = compute_coeffs_mean_std('classic_bias', Noise_batch+CMB_batch.expand(Noise_batch.size()), x=Current_maps[:n_freq])
        if '4' in losses:
            bias_L4, std_L4 = compute_coeffs_mean_std('cross_freq_bias', Noise_batch, x=Current_maps[:n_freq])
        if '6' in losses:
            coeffs_target_L6, std_L6 = compute_coeffs_mean_std('cross_mean', Current_maps[:n_freq].unsqueeze(1).unsqueeze(1).expand((n_freq,n_batch,batch_size,M,N)), cross_contamination_batch=CMB_batch.unsqueeze(0).expand((n_freq,n_batch,batch_size,M,N)))
        if '7' in losses:
            coeffs_target_L7, std_L7 = compute_coeffs_mean_std('cross_mean', Current_maps[:n_freq].unsqueeze(1).unsqueeze(1).expand((n_freq,n_batch,batch_size,M,N)), cross_contamination_batch=Noise_batch)
        
        # Coeffs target computation
        if '1' in losses:
            coeffs_d = wph_op.apply(torch.from_numpy(Mixture), norm=None, pbc=pbc)
            coeffs_target_L1 = torch.cat((torch.unsqueeze(torch.real(coeffs_d) - bias_L1[0],dim=0),torch.unsqueeze(torch.imag(coeffs_d) - bias_L1[1],dim=0)))
        if '4' in losses:
            coeffs_dd = wph_op.apply([torch.from_numpy(Mixture[0]),torch.from_numpy(Mixture[1])], norm=None, cross=True, pbc=pbc)
            coeffs_target_L4 = torch.cat((torch.unsqueeze(torch.real(coeffs_dd) - bias_L4[0],dim=0),torch.unsqueeze(torch.imag(coeffs_dd) - bias_L4[1],dim=0)))
        
        # Mask computation
        if '1' in losses:
            mask_L1 = compute_mask(Current_maps[:n_freq],std_L1)
        if '4' in losses:
            mask_L4 = compute_mask([Current_maps[0],Current_maps[1]],std_L4,cross=True)
        if '6' in losses:
            mask_L6 = compute_mask([Current_maps[:n_freq],Current_maps[2].expand((2,M,N))],std_L6,cross=True)
        if '7' in losses:
            mask_L7 = compute_mask([Current_maps[:n_freq],torch.from_numpy(Noise).to(device)],std_L7,cross=True)
        
        # Minimization
        result = opt.minimize(objective2, Current_maps.cpu().ravel(), method=method, jac=True, tol=None, options=optim_params2)
        final_loss, Current_maps, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        
        # Reshaping
        Current_maps = Current_maps.reshape((n_maps, M, N)).astype(np.float32)
        
        print("Era "+str(i+1)+" done !")
        
    ## Output
    
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    
    if file_name is not None:
        np.save(file_name, [Mixture,Dust,np.array([CMB,CMB]),Noise,Current_maps[:n_freq],np.array([Current_maps[n_freq],Current_maps[n_freq]]),Mixture-Current_maps[:n_freq]-np.array([Current_maps[n_freq],Current_maps[n_freq]]),Current_maps0[:n_freq],Initial_condition])        
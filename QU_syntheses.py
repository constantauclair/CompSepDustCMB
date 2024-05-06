import numpy as np
import argparse
import time
import pywph as pw
import torch
import scipy.optimize as opt

parser = argparse.ArgumentParser()
parser.add_argument('freq', type=int) # 0 for 100, 1 for 143, 2 for 217 or 3 for 353
parser.add_argument('n_syn', type=int)
args = parser.parse_args()
freq = int(args.freq)
n_syn = int(args.n_syn)

freqs = [100,143,217,353]
from_xxx_to_353 = [1/0.0190,1/0.0394,1/0.132,1]

separation = np.load('separation_IQU_B_'+str(freqs[freq])+'_5steps_50iters_Mn=100.npy')[:,64:-64,64:-64].astype(np.float64)*from_xxx_to_353[freq]
I = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_IQU_maps.npy')[0,3,0,64:-64,64:-64].astype(np.float64)
I = (I-np.mean(I))/np.std(I)

Q = separation[2]
U = separation[6]

M, N  =np.shape(Q)
J = 7
L = 4
dn = 5
device = 0
optim_params = {"maxiter": 100}
pbc = False

x_QU = np.array([Q,U])
x_std = x_QU.std(axis=(-1, -2),keepdims=True)
x_mean = x_QU.mean(axis=(-1, -2), keepdims=True)
x_target = (x_QU - x_mean) / x_std

wph_op = pw.WPHOp(M ,N , J, L=L, dn=dn, device=device)
model=["S11","S00","S01","Cphase","C01","C00","L"]
wph_op.load_model(model,tau_grid="exp")
print("Computing stats of target image...")
coeffs_Q = wph_op.apply(x_target[0], norm="auto", pbc=pbc)
wph_op.clear_normalization()
coeffs_U = wph_op.apply(x_target[1], norm="auto", pbc=pbc)
wph_op.clear_normalization()
coeffs_QU = wph_op.apply([x_target[0],x_target[1]], norm="auto", pbc=pbc, cross=True)
wph_op.clear_normalization()
coeffs_IQ = wph_op.apply([I,x_target[0]], norm="auto", pbc=pbc, cross=True)
wph_op.clear_normalization()
coeffs_IU = wph_op.apply([I,x_target[1]], norm="auto", pbc=pbc, cross=True)
print("Done !")

def objective(x):
    start_time = time.time()
    # Reshape x
    x_curr = x.reshape((2, M, N))
    x_curr = torch.from_numpy(x_curr).requires_grad()
    # Compute the loss Q
    loss_tot_Q = torch.zeros(1)
    wph_op.clear_normalization()
    wph_op.apply(x_target[0], norm="auto", pbc=pbc)
    x_curr_Q, nb_chunks = wph_op.preconfigure(x_curr[0], requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_Q, i, norm="auto", ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_Q[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot_Q += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print("Loss Q =",loss_tot_Q.item())
    # Compute the loss U
    loss_tot_U = torch.zeros(1)
    wph_op.clear_normalization()
    wph_op.apply(x_target[1], norm="auto", pbc=pbc)
    x_curr_U, nb_chunks = wph_op.preconfigure(x_curr[1], requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_U, i, norm="auto", ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_U[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot_U += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print("Loss U =",loss_tot_U.item())
    # Compute the loss QU
    loss_tot_QU = torch.zeros(1)
    wph_op.clear_normalization()
    wph_op.apply([x_target[0],x_target[1]], norm="auto", pbc=pbc, cross=True)
    x_curr_QU, nb_chunks = wph_op.preconfigure([x_curr[0],x_curr[1]], requires_grad=True, pbc=pbc, cross=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_QU, i, norm="auto", ret_indices=True, pbc=pbc, cross=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_QU[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot_QU += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print("Loss QU =",loss_tot_QU.item())
    # Compute the loss IQ
    loss_tot_IQ = torch.zeros(1)
    wph_op.clear_normalization()
    wph_op.apply([I,x_target[0]], norm="auto", pbc=pbc, cross=True)
    x_curr_IQ, nb_chunks = wph_op.preconfigure([I,x_curr[0]], requires_grad=True, pbc=pbc, cross=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_IQ, i, norm="auto", ret_indices=True, pbc=pbc, cross=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_IQ[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot_IQ += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print("Loss IQ =",loss_tot_IQ.item())
    # Compute the loss IU
    loss_tot_IU = torch.zeros(1)
    wph_op.clear_normalization()
    wph_op.apply([I,x_target[1]], norm="auto", pbc=pbc, cross=True)
    x_curr_IU, nb_chunks = wph_op.preconfigure([I,x_curr[1]], requires_grad=True, pbc=pbc, cross=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_IU, i, norm="auto", ret_indices=True, pbc=pbc, cross=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_IU[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot_IU += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print("Loss IU =",loss_tot_IU.item())
    # Reshape the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    loss_tot = loss_tot_Q + loss_tot_U + loss_tot_QU + loss_tot_IQ + loss_tot_IU
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    return loss_tot.item(), x_grad.ravel()
    
QU_syntheses=np.zeros([n_syn,2,M,N])

for i in range(n_syn):
    x0 = np.random.normal(0, 1, (2,M,N))
    x0 = torch.from_numpy(x0)
    result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    _, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    x_final = x_final.reshape((2, M, N)).astype(np.float64)
    QU_syntheses[i] = x_final * x_std + x_mean

x_return = np.concatenate((np.array([x_QU]),QU_syntheses),axis=0)  

np.save('QU_synthesis_'+str(freqs[freq])+'.npy',x_return)

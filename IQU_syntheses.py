import sys
import numpy as np
sys.path.append('../Packages/dust_genmodels-main/code/')
from synth import get_initialization, synthesis
from astropy.convolution import convolve, Gaussian2DKernel

separation = np.load('separation_IQU_B_353_5steps_50iters_Mn=100.npy')[:,64:-64,64:-64].astype(np.float64)

def apodize(na, nb, radius):
    na = int(na)
    nb = int(nb)
    ni = int(radius * na)
    nj = int(radius * nb)
    dni = na - ni
    dnj = nb - nj
    tap1d_x = np.zeros(na) + 1.0
    tap1d_y = np.zeros(nb) + 1.0
    tap1d_x[:dni] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
    tap1d_x[na-dni:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
    tap1d_y[:dnj] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
    tap1d_y[nb-dnj:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
    tapper = np.zeros([na,nb])
    for i in range(nb):
        tapper[:,i] = tap1d_x
    for i in range(na):
        tapper[i,:] = tapper[i,:] * tap1d_y
    return tapper

J = 7
L = 4
dn = 5
device = 0
optim_params = {"maxiter": 100}
nsynth = 1
cross_pairs = [[0,1],[0,2],[1,2]]
apo = 0.9

sigma = 1
gauss = Gaussian2DKernel(sigma)

I = convolve(np.log(separation[0]),gauss)
Q = convolve(separation[2],gauss)
U = convolve(separation[6],gauss)

tapper = apodize(np.shape(I)[0],np.shape(I)[1],apo)
I = (I-np.mean(I)) * tapper + np.mean(I)
Q = (Q-np.mean(Q)) * tapper + np.mean(Q)
U = (U-np.mean(U)) * tapper + np.mean(U)

x_IQU = np.array([I,Q,U])
x_std = x_IQU.std(axis=(-1, -2),keepdims=True)
x_mean = x_IQU.mean(axis=(-1, -2), keepdims=True)
x_target = (x_IQU - x_mean) / x_std

x_0 = get_initialization(x_target, nsynth=nsynth)
x_s = synthesis(x_target, x_0, J, L, dn, device=device, optim_params=optim_params, pbc=True, cross_pairs=cross_pairs, wph_model=['S11','S00','S01','Cphase','C00','C01'])

x_syn = x_s * x_std + x_mean

x_final = np.concatenate((np.array([x_IQU]),x_syn),axis=0)  

np.save('IQU_synthesis.npy',x_final)
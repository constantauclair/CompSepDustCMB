import sys
import numpy as np
sys.path.append('../Packages/dust_genmodels-main/code/')
from synth import get_initialization, synthesis
#from astropy.convolution import convolve, Gaussian2DKernel

separation = np.load('separation_IQU_B_353_5steps_50iters_Mn=100.npy')[:,64:-64,64:-64].astype(np.float64)
I = np.load('data/IQU_Planck_data/Sroll20_data/Sroll20_IQU_maps.npy')[0,3,0,64:-64,64:-64].astype(np.float32)

J = 7
L = 4
dn = 5
device = 0
optim_params = {"maxiter": 100}
nsynth = 100
cross_pairs = [[0,1],[0,2],[1,2]]

# sigma = 1
# gauss = Gaussian2DKernel(sigma)

# I = convolve(np.log(separation[0]),gauss)
# Q = convolve(separation[2],gauss)
# U = convolve(separation[6],gauss)

I = np.log(I)
Q = separation[2]
U = separation[6]

x_IQU = np.array([I,Q,U])
x_std = x_IQU.std(axis=(-1, -2),keepdims=True)
x_mean = x_IQU.mean(axis=(-1, -2), keepdims=True)
x_target = (x_IQU - x_mean) / x_std

x_0 = get_initialization(x_target, nsynth=nsynth)
x_s = synthesis(x_target, x_0, J, L, dn, device=device, optim_params=optim_params, pbc=False, cross_pairs=cross_pairs, wph_model=['S11','S00','S01','Cphase','C00','C01','L'])

x_syn = x_s * x_std + x_mean

x_final = np.concatenate((np.array([x_IQU]),x_syn),axis=0)  

np.save('100_IQU_synthesis.npy',x_final)
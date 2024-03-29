import sys
import numpy as np
sys.path.append('../Packages/dust_genmodels-main/code/')
from synth import get_initialization, synthesis

separation = np.load('separation_IQU_B_353_5steps_50iters_Mn=100.npy')[:,64:-64,64:-64].astype(np.float64)

J = 7
L = 4
dn = 5
device = 0
optim_params = {"maxiter": 100}
nsynth = 1
cross_pairs = [[0,1],[0,2],[1,2]]

I = separation[0]
Q = separation[2]
U = separation[6]

x_IQU = np.array([I,Q,U])
#x_IQU[0] = np.log(x_IQU[0])
x_std = x_IQU.std(axis=(-1, -2),keepdims=True)
x_mean = x_IQU.mean(axis=(-1, -2), keepdims=True)
x_IQU = (x_IQU - x_mean) / x_std

x_0 = get_initialization(x_IQU, nsynth=nsynth)
x_s_ = synthesis(x_IQU, x_0, J, L, dn, device=device, optim_params=optim_params, cross_pairs=cross_pairs)

x_s = x_s_ * x_std + x_mean
#x_s[:,0] = np.exp(x_s[:,0])
x_f = x_s    

np.save('IQU_synthesis.npy',x_f)
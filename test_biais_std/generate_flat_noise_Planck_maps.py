import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Center of the sky region
longitude = 300.26
latitude = -16.77
# Resolution (in arcmin)
reso = 2.35
# Map size (in pixel)
N = 512

freq = 143

n_start = 10
n_noise_maps = 100
noise_folder = '../data/IQU_Planck_data/Planck_noise_fits/'

IQU_Noise_maps = np.zeros((3,n_noise_maps,N,N))
for i in range(n_noise_maps):
    noise_map = fits.open(noise_folder+'product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_'+str(freq)+'_full_map_mc_000'+str(1000+n_start+i)[-2:]+'.fits')
    IQU_Noise_maps[0,i] = np.array(hp.gnomview(hp.read_map(noise_map,field=0)*1e6,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    IQU_Noise_maps[1,i] = np.array(hp.gnomview(hp.read_map(noise_map,field=1)*1e6,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    IQU_Noise_maps[2,i] = np.array(hp.gnomview(hp.read_map(noise_map,field=2)*1e6,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    print(i)

np.save("../data/IQU_Planck_data/Chameleon-Musca data/Noise_IQU_"+str(freq)+".npy",IQU_Noise_maps)
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
nside = 2048

freq = 217

n_noise_maps = 100
noise_folder = '../data/IQU_Planck_data/Planck_noise_fits/'

TEB_Noise_maps = np.zeros((3,n_noise_maps,N,N))
for i in range(n_noise_maps):
    noise_map = fits.open(noise_folder+'product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_'+str(freq)+'_full_map_mc_000'+str(1000+i)[-2:]+'.fits')
    IQU_map = [hp.read_map(noise_map,field=0),hp.read_map(noise_map,field=1),hp.read_map(noise_map,field=2)]
    print("Map loaded !")
    TEB_alm = hp.map2alm(IQU_map)
    print("Alm computed !")
    T_map = hp.alm2map(TEB_alm[0],nside)
    E_map = hp.alm2map(TEB_alm[1],nside)
    B_map = hp.alm2map(TEB_alm[2],nside)
    TEB_Noise_maps[0,i] = np.array(hp.gnomview(T_map,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    TEB_Noise_maps[1,i] = np.array(hp.gnomview(E_map,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    TEB_Noise_maps[2,i] = np.array(hp.gnomview(B_map,coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    print(i,"done !")

np.save("../data/IQU_Planck_data/TE correlation data/Noise_TEB_"+str(freq)+".npy",TEB_Noise_maps)
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Center of the sky region
longitude = 300
latitude = -20
# Resolution (in arcmin)
reso = 2.35
# Map size (in pixel)
N = 768
nside = 2048

freq = 100

n_noise_maps = 50
noise_folder = '../data/IQU_Planck_data/Planck_noise_fits/'

TEB_Noise_maps = np.zeros((3,n_noise_maps,N,N))
for i in range(n_noise_maps):
    noise_map = fits.open(noise_folder+'product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_'+str(freq)+'_full_map_mc_000'+str(1000+i+50)[-2:]+'.fits')
    IQU_map = [hp.read_map(noise_map,field=0),hp.read_map(noise_map,field=1),hp.read_map(noise_map,field=2)]
    print("Map "+str(i+1)+" loaded !")
    TEB_Noise_alm = hp.map2alm(IQU_map)
    print("Alm computed !")
    TEB_Noise_map = [hp.alm2map(TEB_Noise_alm[0],2048),hp.alm2map(TEB_Noise_alm[1],2048),hp.alm2map(TEB_Noise_alm[2],2048)]
    TEB_Noise_maps[0,i] = np.array(hp.gnomview(TEB_Noise_map[0],coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    TEB_Noise_maps[1,i] = np.array(hp.gnomview(TEB_Noise_map[1],coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    TEB_Noise_maps[2,i] = np.array(hp.gnomview(TEB_Noise_map[2],coord='G',rot=[longitude,latitude],reso=reso,xsize=N,ysize=N,return_projected_map=True,cmap='inferno'))
    plt.close()
    print("Map "+str(i+1)+" done !")

np.save("../data/IQU_Planck_data/TE correlation data/Noise_TEB_"+str(freq)+"_768_la_suite.npy",TEB_Noise_maps)
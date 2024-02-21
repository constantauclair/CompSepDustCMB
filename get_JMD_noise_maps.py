import numpy as np

n_maps = 100

pols = ['I','Q','U']
freqs = ['100','143','217','353']
missions = ['full','hm1','hm2']

n_pol = len(pols)
n_freq = len(freqs)
n_mission = len(missions)
n_pix = 768

Sroll20_Noise_maps = np.zeros((n_pol,n_freq,n_mission,n_maps,n_pix,n_pix))
for pol in range(n_pol):
    for freq in range(n_freq):
        for mission in range(n_mission):
            for sim in range(n_maps):
                Sroll20_Noise_maps[pol,freq,mission,sim] = np.load('/travail/jdelouis/NEW_NOISE/'+freqs[freq]+'GHz_'+str(sim+100)+'_'+missions[mission]+'_IQU.npy')[pol] * 1e6  
                print('Done !')
         
np.save('data/IQU_Planck_data/TE_correlation_data/Sroll2/Sroll20_Noise_maps.npy',Sroll20_Noise_maps)



























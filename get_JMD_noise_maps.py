import numpy as np

n_maps = 50
N = 768

# 100 GHz
    # TEB
Sroll_Noise_TEB_maps_100_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_100_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_100_HM2_768px = np.zeros((3,n_maps,N,N))
    # IQU
Sroll_Noise_IQU_maps_100_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_100_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_100_HM2_768px = np.zeros((3,n_maps,N,N))

# 143 GHz
    # TEB
Sroll_Noise_TEB_maps_143_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_143_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_143_HM2_768px = np.zeros((3,n_maps,N,N))
    # IQU
Sroll_Noise_IQU_maps_143_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_143_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_143_HM2_768px = np.zeros((3,n_maps,N,N))

# 217 GHz
    # TEB
Sroll_Noise_TEB_maps_217_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_217_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_217_HM2_768px = np.zeros((3,n_maps,N,N))
    # IQU
Sroll_Noise_IQU_maps_217_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_217_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_217_HM2_768px = np.zeros((3,n_maps,N,N))

# 353 GHz
    # TEB
Sroll_Noise_TEB_maps_353_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_353_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_TEB_maps_353_HM2_768px = np.zeros((3,n_maps,N,N))
    # IQU
Sroll_Noise_IQU_maps_353_FM_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_353_HM1_768px = np.zeros((3,n_maps,N,N))
Sroll_Noise_IQU_maps_353_HM2_768px = np.zeros((3,n_maps,N,N))

for i in range(n_maps):
    # 100 GHz
        # TEB
    Sroll_Noise_TEB_maps_100_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_full_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_100_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_hm1_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_100_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_hm2_TEB.npy') * 1e6
        # IQU
    Sroll_Noise_IQU_maps_100_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_full_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_100_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_hm1_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_100_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/100GHz_'+str(i+100)+'_hm2_IQU.npy') * 1e6

    # 143 GHz
        # TEB
    Sroll_Noise_TEB_maps_143_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_full_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_143_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_hm1_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_143_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_hm2_TEB.npy') * 1e6
        # IQU
    Sroll_Noise_IQU_maps_143_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_full_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_143_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_hm1_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_143_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/143GHz_'+str(i+100)+'_hm2_IQU.npy') * 1e6

    # 217 GHz
        # TEB
    Sroll_Noise_TEB_maps_217_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_full_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_217_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_hm1_TEB.npy') * 1e6
    Sroll_Noise_TEB_maps_217_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_hm2_TEB.npy') * 1e6
        # IQU
    Sroll_Noise_IQU_maps_217_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_full_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_217_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_hm1_IQU.npy') * 1e6
    Sroll_Noise_IQU_maps_217_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/217GHz_'+str(i+100)+'_hm2_IQU.npy') * 1e6

    # 353 GHz
    if i < 9:
        # TEB
        Sroll_Noise_TEB_maps_353_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_full_TEB.npy') * 1e6
        Sroll_Noise_TEB_maps_353_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_hm1_TEB.npy') * 1e6
        Sroll_Noise_TEB_maps_353_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_hm2_TEB.npy') * 1e6
        # IQU
        Sroll_Noise_IQU_maps_353_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_full_IQU.npy') * 1e6
        Sroll_Noise_IQU_maps_353_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_hm1_IQU.npy') * 1e6
        Sroll_Noise_IQU_maps_353_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+100)+'_hm2_IQU.npy') * 1e6
    if i >= 9:
        # TEB
        Sroll_Noise_TEB_maps_353_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_full_TEB.npy') * 1e6
        Sroll_Noise_TEB_maps_353_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_hm1_TEB.npy') * 1e6
        Sroll_Noise_TEB_maps_353_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_hm2_TEB.npy') * 1e6
        # IQU
        Sroll_Noise_IQU_maps_353_FM_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_full_IQU.npy') * 1e6
        Sroll_Noise_IQU_maps_353_HM1_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_hm1_IQU.npy') * 1e6
        Sroll_Noise_IQU_maps_353_HM2_768px[:,i] = np.load('/travail/jdelouis/NOISEMAP/353GHz_'+str(i+101)+'_hm2_IQU.npy') * 1e6
         
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_100_FM_768px.npy',Sroll_Noise_TEB_maps_100_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_100_HM1_768px.npy',Sroll_Noise_TEB_maps_100_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_100_HM2_768px.npy',Sroll_Noise_TEB_maps_100_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_100_FM_768px.npy',Sroll_Noise_IQU_maps_100_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_100_HM1_768px.npy',Sroll_Noise_IQU_maps_100_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_100_HM2_768px.npy',Sroll_Noise_IQU_maps_100_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_143_FM_768px.npy',Sroll_Noise_TEB_maps_143_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_143_HM1_768px.npy',Sroll_Noise_TEB_maps_143_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_143_HM2_768px.npy',Sroll_Noise_TEB_maps_143_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_143_FM_768px.npy',Sroll_Noise_IQU_maps_143_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_143_HM1_768px.npy',Sroll_Noise_IQU_maps_143_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_143_HM2_768px.npy',Sroll_Noise_IQU_maps_143_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_217_FM_768px.npy',Sroll_Noise_TEB_maps_217_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_217_HM1_768px.npy',Sroll_Noise_TEB_maps_217_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_217_HM2_768px.npy',Sroll_Noise_TEB_maps_217_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_217_FM_768px.npy',Sroll_Noise_IQU_maps_217_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_217_HM1_768px.npy',Sroll_Noise_IQU_maps_217_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_217_HM2_768px.npy',Sroll_Noise_IQU_maps_217_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_353_FM_768px.npy',Sroll_Noise_TEB_maps_353_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_353_HM1_768px.npy',Sroll_Noise_TEB_maps_353_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_TEB_maps_353_HM2_768px.npy',Sroll_Noise_TEB_maps_353_HM2_768px)

np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_353_FM_768px.npy',Sroll_Noise_IQU_maps_353_FM_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_353_HM1_768px.npy',Sroll_Noise_IQU_maps_353_HM1_768px)
np.save('/data/IQU_Planck_data/TE correlation data/Sroll2/Sroll_Noise_IQU_maps_353_HM2_768px.npy',Sroll_Noise_IQU_maps_353_HM2_768px)



























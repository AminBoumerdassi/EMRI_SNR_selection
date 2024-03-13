""" 
This script does the following things:

1. Samples EMRI parameters
1. Generates EMRIs according to these sampled parameters
2. Calculates the SNRs of these, and plots a histogram of the SNRs."""

use_gpu = True#True#True

#GPU check
import numpy as np
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

#Set up a random number generator
from numpy.random import default_rng
rng = default_rng(seed=2024)

#FEW imports
import sys
import os
import numpy as np
from numpy.random import default_rng
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import SchwarzschildEccentricWaveformBase,FastSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

#Astropy imports for luminosity distance calculations
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import LambdaCDM

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

#Noise generation imports
from lisatools.sensitivity import *

#SNR function imports
from EMRI_SNR_funcs import *

#for SNR calculations
from scipy.signal.windows import tukey

#for visualisation
import matplotlib.pyplot as plt


#Define some EMRI function parameters
no_EMRIs= 50000
T=2#1/12
dt=10
redshift_range= [0.1,0.1]#[0.1, 2]

#Define the Lambda-CDM cosmology used in Ollie's LEMRI paper
''' Source: https://arxiv.org/pdf/2307.06722.pdf'''
Hubb_const= 70.5
Om_matter= 0.274
Om_DE= 0.726
LCDM= LambdaCDM(H0= Hubb_const, Om0= Om_matter, Ode0= Om_DE)

#Sample a batch of parameters, initialise the SNR array
params= sample_EMRI_parameters(LCDM, batch_size=no_EMRIs, redshift_range=redshift_range)
SNR_arr= np.zeros(params.shape[0])

#Initialise the TDI wrapper
#Then iterate the waveform generation and SNR calcualation
EMRI_TDI_0PA_ecc= init_EMRI_TDI(T=T, dt=dt, use_gpu=use_gpu)
for i in range(params.shape[0]):
    waveform= generate_EMRI_AET(params[i], EMRI_TDI_0PA_ecc)
    SNR_arr[i]= SNR_AET(waveform)

    
#Save all the sampled parameters and respective SNRs
np.save("EMRI_params.npy", params)
np.save("EMRI_SNRs.npy", SNR_arr)
    
#Plot a histogram of the EMRI SNRs
''' I think this becomes very time-consuming for very large datasets'''
plt.hist(SNR_arr, bins=int(no_EMRIs/100))
plt.title("{:} EMRI SNRs with observation window {:} years".format(no_EMRIs,T))
plt.xlabel("SNR")
plt.ylabel("frequency")
plt.savefig("SNR_hist.png")



''' Benchmarks:
This code took about 3 hours to process 50k EMRIs of duration 2 years.

Further time savings could be made by:

1. Not plotting histograms in this script as it becomes very slow when dealing with large datasets.

2. If we really want to plot a histogram, use the xp function for histograms.

3. It may make more sense to move the histogram plotting onto a separate script for the sake of saving time.


'''

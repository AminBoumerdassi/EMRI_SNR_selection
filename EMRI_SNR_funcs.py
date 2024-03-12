''' 
This file contains GPU-enabled functions for:
    1. Sampling EMRI parameters
    2. Generating h(t) according to the Schwarzchild ecc. flux model
    3. Utility functions 
'''


def noise_td_AET(N, dt, channel="A", use_gpu=True):
    '''
    Generate noise in the time-domain in the AET channels using
    their respective PSDs.
    '''
    if use_gpu:
        xp=cp
    else:
        xp=np
        
    if channel=="A" or channel=="E":
        channel= "AE"
        
    #Extract frequency bins for use in PSD
    freq = xp.fft.rfftfreq(N , dt)
    PSD = get_sensitivity(freq, sens_fn="noisepsd_"+channel, return_type="PSD")#This can result in NaNs for noise_f
    PSD[0]=PSD[1]#Changing the NaN in PSD[0]
    #Draw samples from multivariate Gaussian
    variance_noise_f= N*PSD/(4*dt)
    noise_f = xp.random.normal(0,np.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,np.sqrt(variance_noise_f))
    #Transforming the frequency domain signal into the time domain
    return xp.fft.irfft(noise_f, n=N)

def zero_pad(data, use_gpu=True):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    """
    if use_gpu:
        xp=cp
    else:
        xp=np
    
    N = len(data)
    pow_2 = xp.ceil(xp.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD, use_gpu=True):

    if use_gpu:
        xp=cp
    else:
        xp=np

    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)

    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))

def SNR_AET(waveform, dt=10, use_gpu=True):
    ''' waveform_AET should have shape (no. channels, length timeseries)'''
    
    if use_gpu:
        xp=cp
    else:
        xp=np
    
    window = xp.asarray(tukey(len(waveform[0]),0.01))

    TDI_channels = ['TDIA','TDIE','TDIT']
    N_channels = waveform.shape[0]

    EMRI_AET_1PA_w_pad = [zero_pad(window*(waveform[i]), use_gpu=use_gpu) for i in range(N_channels)]
    N_t = len(EMRI_AET_1PA_w_pad[0])

    EMRI_AET_1PA_fft = xp.asarray([xp.fft.rfft(item) for item in EMRI_AET_1PA_w_pad])
    freq = xp.fft.rfftfreq(N_t,dt)
    freq[0] = freq[1]   # To "retain" the zeroth frequency

    # Define PSDs
    PSD_AET = [noisepsd_AE2(freq),noisepsd_AE2(freq),noisepsd_T(freq)]
    SNR2_AET = xp.asarray([inner_prod(EMRI_AET_1PA_fft[i],EMRI_AET_1PA_fft[i],N_t,dt,PSD_AET[i], use_gpu=use_gpu) for i in range(N_channels)])

    #Sum the SNRs in quadrature
    return xp.sum(SNR2_AET)**(1/2)

def sample_EMRI_parameters(LCDM_model, batch_size=1, fixed_redshift=True):
    ''' 
    Generate data according to randomised parameters.
    
    The parameters are:
    
    eta: mass ratio mu/M
    M: larger BH mass
    mu: smaller CO mass
    e0: initial eccentricity
    p0: Initial semilatus rectum
    theta: Polar viewing angle
    phi: Azimuthal viewing angle
    phi_phi0: Initial phase of azimuthal viewing angle in phi plane
    phi_r0: Initial phase of azimuthal viewing angle is r plane
    dist: luminosity distance
    qS: Sky location polar angle in ecliptic coordinates
    phiS: Sky location azimuthal angle in ecliptic coordinates
    qK: Initial BH spin polar angle in ecliptic coordinates
    phiK: Initial BH spin azimuthal angle in ecliptic coordinates
    
    LDCM model used to convert redshifts to luminosity distances.
    '''
    set_of_eta= rng.uniform(1e-6,1e-4, size= batch_size)#Used indirectly in calculating mu
    set_of_M= rng.uniform(1e4, 1e7, size = batch_size)
    set_of_mu= set_of_eta*set_of_M
    set_of_a= np.array([None for i in range(batch_size)])
    set_of_e0= rng.uniform(0, .7, size = batch_size)#Initial eccentricity
    set_of_p0= rng.uniform(10, 16+set_of_e0)#Needs to be based on set_of_e0
    set_of_x0= np.array([None for i in range(batch_size)])
    
    if fixed_redshift==True:
        #redshift 0.1 is roughly 0.5Gpc in lum. distance
        #redshift 0.2 is roughly 1Gpc in lum. distance
        set_of_redshift= 0.1*np.ones(batch_size) * cu.redshift
    else:
        set_of_redshift= rng.uniform(0.1, 2, size= batch_size) * cu.redshift    
    set_of_dist = np.array(set_of_redshift.to(u.Gpc, cu.redshift_distance(LCDM_model, kind="luminosity")))
    
    set_of_theta= rng.uniform(-np.pi/2, +np.pi/2, size = batch_size)#polar viewing angle
    set_of_phi= rng.uniform(0, 2*np.pi, size = batch_size)
    set_of_phi_phi0= rng.uniform(0, 2*np.pi, size = batch_size)
    set_of_phi_theta0= rng.uniform(0, 2*np.pi, size = batch_size)
    set_of_phi_r0= rng.uniform(0, 2*np.pi, size = batch_size)
    
    
    #TDI parameters
    #Check these parameter ranges - they may not be correct
    set_of_qS= rng.uniform(-np.pi/2, +np.pi/2, size=batch_size)
    set_of_phiS= rng.uniform(0, 2*np.pi, size=batch_size)
    set_of_qK= rng.uniform(-np.pi/2, +np.pi/2, size=batch_size)
    set_of_phiK= rng.uniform(0, 2*np.pi, size=batch_size)
        
    return np.vstack((set_of_M, set_of_mu, set_of_a,
                     set_of_p0,set_of_e0, set_of_x0,
                     set_of_dist, set_of_qS,set_of_phiS,
                     set_of_qK, set_of_phiK, set_of_phi_phi0,
                     set_of_phi_theta0, set_of_phi_r0)).T


def init_EMRI_TDI(T=3, dt=10, use_gpu=True):
    '''
    Initialises the fastLISArespose TDI package.
    '''
    if use_gpu:
        xp=cp
    else:
        xp=np

    # order of the langrangian interpolation
    t0 = 20000.0   # How many samples to remove from start and end of simulations
    order = 25

    orbit_file_esa = "/nesi/project/uoa00195/software/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"

    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa) # these are the orbit files that you will have cloned if you are using Michaels code.
    # you do not need to generate them yourself. They’re already generated. 

    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET"
        )

    TDI_channels = ['TDIA','TDIE','TDIT']
    N_channels = len(TDI_channels)
    waveform_kwargs={"sum_kwargs":{"pad_output":True}}


    generic_class_waveform_0PA_ecc = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu = use_gpu, **waveform_kwargs)

    # ================== APPLY THE RESPONSE =================================

    index_lambda = 7 # Index of polar angle
    index_beta = 8   # Index of phi angle

    EMRI_TDI_0PA_ecc = ResponseWrapper(generic_class_waveform_0PA_ecc,T,dt,
                              index_lambda,index_beta,t0=t0,
                              flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                              remove_garbage = "zero", n_overide= int(np.ceil(YRSID_SI*T/dt)), **tdi_kwargs_esa)
    return EMRI_TDI_0PA_ecc

def generate_EMRI_AET(params, EMRI_TDI_0PA_ecc):
    ''' 
    Generate a time-domain EMRI according to the Fast Schwarzchild ecc. flux model.
    
    Parameters are:
    
    [M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    
    For an SNR 20 EMRI, 
    use params = [1e6, 40, 0.92, 15.0975, 0.2, np.cos(1), 8.89, 0.3, 0.3, 0.8, 1, 1, 2, 3]
    '''
    waveform = EMRI_TDI_0PA_ecc(*params)
    return xp.array(waveform)



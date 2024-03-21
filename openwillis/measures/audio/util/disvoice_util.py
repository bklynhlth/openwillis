"""Functions below are taken from disvoice package
https://github.com/jcvasquezc/DisVoice/tree/master
and are modified for efficiency and speed
"""
# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

import sys

import numpy as np
from scipy.io.wavfile import read
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks

from disvoice.glottal.GCI import iaif, compute_h1h2_hrf_frame, find_amid_t
from disvoice.glottal.utils_gci import create_continuous_smooth_f0, GetLPCresidual, get_MBS, get_MBS_GCI_intervals, search_res_interval_peaks
import pysptk


# Parameters for glottal source parameter estimation
T0_num=3 #Number of local glottal pulses to be used for harmonic spectrum
min_harm_num=5
HRF_freq_max=5000 # Maximum frequency used for harmonic measurement
qoq_level=0.5 # threhold for QOQ estimation
F0min=20
F0max=500

def get_costm_matrix(GCI_N, x, trans_wgt, ncands, pulseLen, n):
    """
    ------------------------------------------------------------------------------------------------------

    This function is meant to replace the get_costm_matrix function in disvoice package at
    https://github.com/jcvasquezc/DisVoice/blob/master/disvoice/glottal/utils_gci.py#L309
    Calculates the cost matrix for the dynamic programming algorithm.

    Parameters:
    ...........
    GCI_N : numpy array
        Matrix containing N by M candidate GCI locations (in samples)
    x : numpy array
        Speech signal
    trans_wgt: float
        Transition cost weight
    ncands: int
        Number of candidates
    pulseLen: int
        Length of the pulses
    n: int
        Current frame
    
    Returns:
    ...........
    costm: numpy array
        Cost matrix for the dynamic programming algorithm

    ------------------------------------------------------------------------------------------------------
    """
    # Initialize the cost matrix
    costm = np.zeros((ncands, ncands))

    # Extract the pulses for the current and previous frames
    pulses_cur = [x[max(0, int(GCI_N[n, c] - pulseLen / 2)):min(int(GCI_N[n, c] + pulseLen / 2), len(x))] for c in range(ncands)]
    pulses_prev = [x[max(1, int(GCI_N[n - 1, p] - pulseLen / 2)):min(len(x), int(GCI_N[n - 1, p] + pulseLen / 2))] for p in range(ncands)]

    # Make sure all pulses are the same length by padding with NaNs
    max_len = max(max(len(pulse) for pulse in pulses_cur), max(len(pulse) for pulse in pulses_prev))
    pulses_cur = [np.concatenate((pulse, np.full(max_len - len(pulse), np.nan))) for pulse in pulses_cur]
    pulses_prev = [np.concatenate((pulse, np.full(max_len - len(pulse), np.nan))) for pulse in pulses_prev]

    # Convert to NumPy arrays for vectorized computation
    pulses_cur = np.array(pulses_cur)
    pulses_prev = np.array(pulses_prev)

    # Compute the correlations using NumPy functions
    mean_cur = np.nanmean(pulses_cur, axis=1, keepdims=True)
    mean_prev = np.nanmean(pulses_prev, axis=1, keepdims=True)
    std_cur = np.nanstd(pulses_cur, axis=1, keepdims=True)
    std_prev = np.nanstd(pulses_prev, axis=1, keepdims=True)
    corr_matrix = ((pulses_cur - mean_cur) @ (pulses_prev - mean_prev).T) / (max_len * std_cur * std_prev.T)

    # Apply conditions and scaling
    conditions = ((std_cur > 0) & (std_prev.T > 0)).squeeze()
    costm = np.where(conditions, (1 - np.abs(corr_matrix)) * trans_wgt, costm)

    return costm

def RESON_dyProg_mat(GCI_relAmp,GCI_N,F0mean,x,fs,trans_wgt,relAmp_wgt):
    # Function to carry out dynamic programming method described in Ney (1989)
    # and used previously in the ESPS GCI detection algorithm. The method
    # considers target costs and transition costs which are accumulated in
    # order to select the `cheapest' path, considering previous context

    # USAGE: INPUT
    #        GCI_relAmp - target cost matrix with N rows (GCI candidates) by M
    #                     columns (mean based signal derived intervals).
    #        GCI_N      - matrix containing N by M candidate GCI locations (in
    #                     samples)
    #        F0_inter   - F0 values updated every sample point
    #        x          - speech signal
    #        fs         - sampling frequency
    #        trans_wgt  - transition cost weight
    #
    #        OUTPUT
    #        GCI        - estimated glottal closure instants (in samples)
    # =========================================================================
    # === FUNCTION CODED BY JOHN KANE AT THE PHONETICS LAB TRINITY COLLEGE ====
    # === DUBLIN. 25TH October 2011 ===========================================
    # =========================================================================

    # =========================================================================
    # === FUNCTION ADAPTED AND CODED IN PYTHON BY J. C. Vasquez-Correa
    #   AT THE PATTERN RECOGNITION LAB, UNIVERSITY OF ERLANGEN-NUREMBERG ====
    # === ERLANGEN, MAY, 2018 ===========================================
    # =========================================================================



    ## Initial settings

    GCI_relAmp=np.asarray(GCI_relAmp)
    relAmp_wgt=np.asarray(relAmp_wgt)
    cost = GCI_relAmp*relAmp_wgt
    #print(cost.shape)
    GCI_N=np.asarray(GCI_N)
    ncands=GCI_N.shape[1]
    nframe=GCI_N.shape[0]
    #print(ncands, nframe, cost.shape)
    prev=np.zeros((nframe,ncands))
    pulseLen = int(fs/F0mean)

    for n in range(1,nframe):
        costm = get_costm_matrix(GCI_N, x, trans_wgt, ncands, pulseLen, n)
        costm=costm+np.tile(cost[n-1,0:ncands],(ncands,1))
        costm=np.asarray(costm)
        costi=np.min(costm,0)
        previ=np.argmin(costm,0)
        cost[n,0:ncands]=cost[n,0:ncands]+costi
        prev[n,0:ncands]=previ

    best=np.zeros(n+1)
    best[n]=np.argmin(cost[n,0:ncands])



    for i in range(n-1,1,-1):

        best[i-1]=prev[i,int(best[i])]

    GCI_opt=np.asarray([GCI_N[n,int(best[n])] for n in range(nframe)])

    return GCI_opt

def se_vq_varf0(x,fs, f0=None):
    """
    Function to extract GCIs using an adapted version of the SEDREAMS 
    algorithm which is optimised for non-modal voice qualities (SE-VQ). Ncand maximum
    peaks are selected from the LP-residual signal in the interval defined by
    the mean-based signal. 
    
    A dynamic programming algorithm is then used to select the optimal path of GCI locations. 
    Then a post-processing method, using the output of a resonator applied to the residual signal, is
    carried out to remove false positives occurring in creaky speech regions.
    
    Note that this method is slightly different from the standard SE-VQ
    algorithm as the mean based signal is calculated using a variable window
    length. 
    
    This is set using an f0 contour interpolated over unvoiced
    regions and heavily smoothed. This is particularly useful for speech
    involving large f0 excursions (i.e. very expressive speech).

    :param x:  speech signal (in samples)
    :param fs: sampling frequency (Hz)
    :param f0: f0 contour (optional), otherwise its computed  using the RAPT algorithm
    :returns: GCI Glottal closure instants (in samples)
    
    References:
          Kane, J., Gobl, C., (2013) `Evaluation of glottal closure instant 
          detection in a range of voice qualities', Speech Communication
          55(2), pp. 295-314.
    

    ORIGINAL FUNCTION WAS CODED BY JOHN KANE AT THE PHONETICS AND SPEECH LAB IN 
    TRINITY COLLEGE DUBLIN ON 2013.
    
    THE SEDREAMS FUNCTION WAS CODED BY THOMAS DRUGMAN OF THE UNIVERSITY OF MONS
   
    THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBER- GERMANY
    AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    JCAMILO.VASQUEZ@UDEA.EDU.CO
    https//jcvasquezc.github.io
    """
    if f0 is None:
        f0 = []
    if len(f0)==0 or sum(f0)==0:
        size_stepS=0.01*fs
        voice_bias=-0.2
        x=x-np.mean(x)
        x=x/np.max(np.abs(x))
        data_audiof=np.asarray(x*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=F0min, max=F0max, voice_bias=voice_bias, otype='f0')


    F0nz=np.where(f0>0)[0]
    F0mean=np.median(f0[F0nz])
    VUV=np.zeros(len(f0))
    VUV[F0nz]=1
    if F0mean<70:
        F0mean=80

    # Interpolate f0 over unvoiced regions and heavily smooth the contour
    ptos=np.linspace(0,len(x),len(VUV))
    VUV_inter=np.interp(np.arange(len(x)), ptos, VUV)
    VUV_inter[np.where(VUV_inter>0.5)[0]]=1
    VUV_inter[np.where(VUV_inter<=0.5)[0]]=0
    f0_int, f0_samp=create_continuous_smooth_f0(f0,VUV,x)
    T0mean = fs/f0_samp
    winLen = 25 # window length in ms
    winShift = 5 # window shift in ms
    LPC_ord = int((fs/1000)+2) # LPC order
    Ncand=5 # Number of candidate GCI residual peaks to be considered in the dynamic programming
    trans_wgt=1 # Transition cost weight
    relAmp_wgt=0.3 # Local cost weight

    #Calculate LP-residual and extract N maxima per mean-based signal determined intervals
    res = GetLPCresidual(x,winLen*fs/1000,winShift*fs/1000,LPC_ord, VUV_inter) # Get LP residual
    MBS = get_MBS(x,fs,T0mean) # Extract mean based signal
    interval = get_MBS_GCI_intervals(MBS,fs,T0mean,F0max) # Define search intervals
    [GCI_N,GCI_relAmp] = search_res_interval_peaks(res,interval,Ncand, VUV_inter) # Find residual peaks
    if len(np.asarray(GCI_N).shape) > 1:
        GCI = RESON_dyProg_mat(GCI_relAmp,GCI_N,F0mean,x,fs,trans_wgt,relAmp_wgt) # Do dynamic programming
    else:
        GCI = None

    return GCI

def extract_glottal_signal(x, fs):
    """Extract the glottal flow and the glottal flow derivative signals

    :param x: data from the speech signal.
    :param fs: sampling frequency
    :returns: glottal signal
    :returns: derivative  of the glottal signal
    :returns: glottal closure instants

    >>> from scipy.io.wavfile import read
    >>> glottal=Glottal()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> fs, data_audio=read(audio)
    >>> glottal, g_iaif, GCIs=glottal.extract_glottal_signal(data_audio, fs)

    """
    winlen = int(0.025*fs)
    winshift = int(0.005*fs)
    x = x-np.mean(x)
    x = x/float(np.max(np.abs(x)))
    GCIs = se_vq_varf0(x, fs)
    g_iaif = np.zeros(len(x))
    glottal = np.zeros(len(x))

    if GCIs is None:
        sys.warn("not enought voiced segments were found to compute GCI")
        return glottal, g_iaif, GCIs

    start = 0
    stop = int(start+winlen)
    win = np.hanning(winlen)

    while stop <= len(x):

        x_frame = x[start:stop]
        pGCIt = np.where((GCIs > start) & (GCIs < stop))[0]
        GCIt = GCIs[pGCIt]-start

        g_iaif_f = iaif(x_frame, fs, GCIt)
        glottal_f = cumtrapz(g_iaif_f, dx=1/fs)
        glottal_f = np.hstack((glottal[start], glottal_f))
        g_iaif[start:stop] = g_iaif[start:stop]+g_iaif_f*win
        glottal[start:stop] = glottal[start:stop]+glottal_f*win
        start = start+winshift
        stop = start+winlen
    g_iaif = g_iaif-np.mean(g_iaif)
    g_iaif = g_iaif/max(abs(g_iaif))

    glottal = glottal-np.mean(glottal)
    glottal = glottal/max(abs(glottal))
    glottal = glottal-np.mean(glottal)
    glottal = glottal/max(abs(glottal))

    return glottal, g_iaif, GCIs

def get_vq_params(gf, gfd, fs, GCI):
    """
    Function to estimate the glottal parameters: NAQ, QOQ, H1-H2, and HRF

    This function can be used to estimate a range of conventional glottal
    source parameters often used in the literature. This includes: the
    normalized amplitude quotient (NAQ), the quasi-open quotient (QOQ), the
    difference in amplitude of the first two harmonics of the differentiated
    glottal source spectrum (H1-H2), and the harmonic richness factor (HRF)
    
    :param gf: [samples] [N] Glottal flow estimation
    :param gfd: [samples] [N] Glottal flow derivative estimation
    :param fs: [Hz] [1] sampling frequency
    :param GCI: [samples] [M] Glottal closure instants
    :returns: NAQ [s,samples] [Mx2] Normalised amplitude quotient
    :returns: QOQ[s,samples] [Mx2] Quasi-open quotient
    :returns: HRF[s,samples] [Mx2] Harmonic richness factor
    
    References:
     [1] Alku, P., B ackstrom, T., and Vilkman, E. Normalized amplitude quotient for parameterization of the glottal flow. Journal of the Acoustical Society of America, 112(2):701-710, 2002.
     
     [2] Hacki, T. Klassifizierung von glottisdysfunktionen mit hilfe der elektroglottographie. Folia Phoniatrica, pages 43-48, 1989.
     
     [3] Alku, P., Strik, H., and Vilkman, E. Parabolic spectral parameter - A new method for quantification of the glottal flow. Speech Communication, 22(1):67-79, 1997.
     
     [4] Hanson, H. M. Glottal characteristics of female speakers: Acoustic correlates. Journal of the Acoustical Society of America, 10(1):466-481, 1997.
        
     [5] Childers, D. G. and Lee, C. K. Voice quality factors: Analysis, synthesis and perception. Journal of the Acoustical Society of  America, 90(5):2394-2410, 1991.
    
    Function Coded by John Kane @ The Phonetics and Speech Lab
    Trinity College Dublin, August 2012

    THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBERGER- GERMANY
    AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    JCAMILO.VASQUEZ@UDEA.EDU.CO
    https//jcvasquezc.github.io
    """


    NAQ=np.zeros(len(GCI))
    QOQ=np.zeros(len(GCI))
    H1H2=np.zeros(len(GCI))
    HRF=np.zeros(len(GCI))
    T1=np.zeros(len(GCI))
    T2=np.zeros(len(GCI))
    glot_shift=np.round(0.5/1000*fs)
    
    if len(GCI) <= 1:
        sys.warn("not enough voiced segments were found to compute GCI")
        return NAQ, QOQ, HRF
    start=0
    stop=int(GCI[0])
    T0=GCI[1]-GCI[0]

    for n in range(len(GCI)):
        # Get glottal pulse compensated for zero-line drift
        if n>0:
            start=int(GCI[n-1])
            stop=int(GCI[n])
            T0=GCI[n]-GCI[n-1]
            if T0==0 and n>=2:
                T0=GCI[n]-GCI[n-2]
                start=int(GCI[n-2])
        F0=fs/T0

        if T0<=0 or F0<=F0min or F0>=F0max:
            continue

        gf_comb=[gf[start], gf[stop]]
        line=0
        if start!=stop and len(gf_comb)>1:
            line=np.interp(np.arange(stop-start), np.linspace(0,stop-start,2), gf_comb)
        elif start!=stop and len(gf_comb)<=1:
            line=gf_comb
        gf_seg=gf[start:stop]
        gf_seg_comp=gf_seg-line
        f_ac=np.max(gf_seg_comp)
        Amid=f_ac*qoq_level
        max_idx=np.argmax(gf_seg_comp)
        T1[n],T2[n] = find_amid_t(gf_seg_comp,Amid,max_idx)

        if stop+glot_shift<=len(gfd):
            stop=int(stop+glot_shift)
        gfd_seg=gfd[start:stop]

        # get NAQ and QOQ
        d_peak=np.max(np.abs(gfd_seg))
        
        NAQ[n]=(f_ac/d_peak)/T0
        QOQ[n]=(T2[n]-T1[n])/T0
        # Get frame positions for H1-H2 parameter
        _, HRF[n]=compute_h1h2_hrf_frame(GCI[n], T0, T0_num, gfd, F0, fs)

    return NAQ, QOQ, HRF

def peakdetect(y_axis, lookahead=200, delta=0):
    """
    ------------------------------------------------------------------------------------------------------

    This function is meant to replace the peakdetect function in disvoice package at
    https://github.com/jcvasquezc/DisVoice/blob/master/disvoice/glottal/peakdetect.py#L131
    Calculates the peaks and valleys of a signal and returns them in two lists.

    Parameters:
    ...........
    y_axis : list or numpy array
        Contains the signal to be processed
    lookahead : int, optional (default = 200)
        distance to look ahead from a peak candidate to determine if it is the actual peak
    delta : float, optional (default = 0)
        this specifies a minimum difference between a peak and the following points,
         before a peak may be considered a peak.
    
    Returns:
    ...........
    max_peaks, min_peaks : list
        Lists of maximas and minimas respectively. Each item of the list is a tuple: (position, value)

    ------------------------------------------------------------------------------------------------------
    """
    max_peaks, _ = find_peaks(y_axis[:-lookahead])
    min_peaks, _ = find_peaks(-y_axis[:-lookahead])

    return [(x, y_axis[x]) for x in max_peaks], [(x, y_axis[x]) for x in min_peaks]

def extract_features_file(audio):
    """Extract the glottal features from an audio file

    :param audio: .wav audio file.
    :returns: features computed from the audio file.

    >>> glottal=Glottal()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features1=glottal.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=glottal.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=glottal.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> glottal.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
    """

    if audio.find('.wav') == -1 and audio.find('.WAV') == -1:
        raise ValueError(audio+" is not a valid wav file")
    fs, data_audio = read(audio)

    if len(data_audio.shape)>1:
        data_audio = data_audio.mean(1)

    data_audio = data_audio-np.mean(data_audio)
    data_audio = data_audio/float(np.max(np.abs(data_audio)))

    size_frame = 0.2
    size_step = 0.05

    size_frameS = size_frame*float(fs)
    size_stepS = size_step*float(fs)
    overlap = size_stepS/size_frameS
    nF = int((len(data_audio)/size_frameS/overlap))-1
    data_audiof = np.asarray(data_audio*(2**15), dtype=np.float32)
    f0 = pysptk.sptk.rapt(data_audiof, fs, int(
        0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
    sizef0 = int(size_frame/0.01)
    stepf0 = int(size_step/0.01)
    startf0 = 0
    stopf0 = sizef0

    glottal, g_iaif, GCI = extract_glottal_signal(data_audio, fs)

    avgNAQt = np.zeros(nF)
    varNAQt = np.zeros(nF)
    avgQOQt = np.zeros(nF)
    varQOQt = np.zeros(nF)
    avgHRFt = np.zeros(nF)
    varHRFt = np.zeros(nF)

    rmwin = []
    for l in range(nF):
        init = int(l*size_stepS)
        endi = int(l*size_stepS+size_frameS)
        gframe = glottal[init:endi]
        dgframe = glottal[init:endi]
        pGCIt = np.where((GCI > init) & (GCI < endi))[0]
        gci_s = GCI[pGCIt]-init
        f0_frame = f0[startf0:stopf0]
        pf0framez = np.where(f0_frame != 0)[0]
        f0nzframe = f0_frame[pf0framez]
        if len(f0nzframe) < 5:
            startf0 += stepf0
            stopf0 += stepf0
            rmwin.append(l)
            continue

        startf0 += stepf0
        stopf0 += stepf0
        NAQ, QOQ, HRF = get_vq_params(
            gframe, dgframe, fs, gci_s)
        avgNAQt[l] = np.mean(NAQ)
        varNAQt[l] = np.std(NAQ)
        avgQOQt[l] = np.mean(QOQ)
        varQOQt[l] = np.std(QOQ)
        avgHRFt[l] = np.mean(HRF)
        varHRFt[l] = np.std(HRF)

    if len(rmwin) > 0:
        avgNAQt = np.delete(avgNAQt, rmwin)
        varNAQt = np.delete(varNAQt, rmwin)
        avgQOQt = np.delete(avgQOQt, rmwin)
        varQOQt = np.delete(varQOQt, rmwin)
        avgHRFt = np.delete(avgHRFt, rmwin)
        varHRFt = np.delete(varHRFt, rmwin)

    feat = np.stack((avgHRFt, varHRFt, avgNAQt, varNAQt, avgQOQt, varQOQt), axis=1)

    return feat.mean(0).tolist()

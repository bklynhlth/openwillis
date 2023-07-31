"""Functions below are taken from disvoice package
https://github.com/jcvasquezc/DisVoice/tree/master
and are modified for efficiency and speed
"""
import sys

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import find_peaks

from disvoice.glottal import Glottal
import pysptk


# Parameters for glottal source parameter estimation
T0_num=3 #Number of local glottal pulses to be used for harmonic spectrum
min_harm_num=5
HRF_freq_max=5000 # Maximum frequency used for harmonic measurement
qoq_level=0.5 # threhold for QOQ estimation
F0min=20
F0max=500

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

def compute_h1h2_hrf_frame(GCIn, T0, T0_num, gfd, F0, fs):
    H1H2=0
    HRF=0

    if GCIn-int((T0*T0_num)/2)>0:
        f_start=int(GCIn-int((T0*T0_num)/2))
    else:
        f_start=0
    if GCIn+int((T0*T0_num)/2)<=len(gfd):
        f_stop=int(GCIn+int((T0*T0_num)/2))
    else:
        f_stop=len(gfd)
    f_frame=gfd[f_start:f_stop]
    f_win=f_frame*np.hamming(len(f_frame))
    f_spec=20*np.log10(np.abs(np.fft.fft(f_win, fs)))

    f_spec=f_spec[0:int(len(f_spec)/2)]
    # get H1-H2 and HRF
    [max_peaks, min_peaks]=peakdetect(f_spec,lookahead = int(T0))


    if len(max_peaks)==0:
        return 0, 0
    h_idx, h_amp=zip(*max_peaks)
    HRF_harm_num=np.fix(HRF_freq_max/F0)
    if len(h_idx)>=min_harm_num:
        temp1=np.arange(HRF_harm_num)*F0
        f0_idx=np.zeros(len(h_idx))
        for mp in range(len(h_idx)):

            temp2=h_idx[mp]-temp1
            temp2=np.abs(temp2)
            posmin=np.where(temp2==min(temp2))[0]
            if len(posmin)>1:
                posmin=posmin[0]

            if posmin<len(h_idx):
                f0_idx[mp]=posmin
            else:
                f0_idx[mp]=len(h_idx)-1

        f0_idx=[int(mm) for mm in f0_idx]

        H1H2=h_amp[f0_idx[0]]-h_amp[f0_idx[1]]
        harms=[h_amp[mm] for mm in f0_idx[1:]]
        HRF=sum(harms)/h_amp[f0_idx[0]]

    return H1H2, HRF

def find_amid_t(glot_adj, Amid, Tz):
    #Function to find the start and stop positions of the quasi-open phase.
    T1=0
    T2=0
    if Tz!=0:
        n=Tz
        while glot_adj[n]>Amid and n>2:
            n=n-1
        T1=n
        n=Tz
        while glot_adj[n] > Amid and n < len(glot_adj)-1:
            n=n+1
        T2=n
    return T1, T2

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

    glottal_obj = Glottal()

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

    glottal, g_iaif, GCI = glottal_obj.extract_glottal_signal(data_audio, fs)

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

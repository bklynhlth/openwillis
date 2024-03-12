"""Common functions for acoustic analysis."""
# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

import os
import json
import math

import numpy as np
import pandas as pd
import scipy
import numpy.matlib

from parselmouth import Sound
from parselmouth.praat import call
from pydub import AudioSegment, silence
import librosa


def common_summary(df, col_name):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates common summary statistics for a given column of a dataframe.

    Parameters:
    ...........
    df : pandas dataframe
        the dataframe to summarize.
    col_name : str
        the name of the column to summarize.

    Returns:
    ...........
    df_summ : pandas dataframe
        a dataframe containing the summary statistics of the given column.

    ------------------------------------------------------------------------------------------------------
    """
    mean = df.mean()
    std = df.std()
    min_val = df.min()
    max_val = df.max()
    range_val = max_val - min_val

    values = [mean, std, range_val]
    cols = [col_name + '_mean', col_name + '_stddev', col_name + '_range']

    df_summ = pd.DataFrame([values], columns= cols)
    return df_summ

def silence_summary(sound, df, measures):
    """
    ------------------------------------------------------------------------------------------------------
    Calculates silence summary statistics for a given audio file.

    Parameters:
    ...........
    sound : Praat sound object;
        the audio file to analyze.
    df : pandas dataframe
        the dataframe containing the silence intervals in the audio file.
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    silence_summ : pandas dataframe
        a dataframe containing the summary statistics of the silence intervals.

    ------------------------------------------------------------------------------------------------------
    """
    duration = call(sound, "Get total duration")
    df_silence = df[measures['voicesilence']]

    if len(df_silence) == 0:
        spir = 0
        dur_med = np.NaN
        dur_mad = np.NaN
    else:
        # inappropriate pauses stats
        df2 = df[df_silence > 0.05][df_silence < 2]

        total_pause_time = df_silence.sum() # in seconds
        total_speech_time = duration - total_pause_time # in seconds
        spir = len(df2) / total_speech_time

        dur_med = df2[measures['voicesilence']].median()

        dur_mad = np.median(np.abs(df2[measures['voicesilence']] - df2[measures['voicesilence']].mean()))

    cols = [
        measures['spir'], measures['pause_meddur'], measures['pause_maddur']    
    ]

    silence_summ = pd.DataFrame(
        [[spir, dur_med, dur_mad]], columns = cols
    )
    return silence_summ

def voice_frame(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the percentage of frames in the given audio file that contain voice.

    Parameters:
    ...........
    sound : Praat sound object
        the audio file to analyze.
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    df : pandas dataframe;
        a dataframe containing the percentage of frames in the audio file that contain voice.

    ------------------------------------------------------------------------------------------------------
    """
    pitch = call(sound, "To Pitch", 0.0, 75, 500)

    total_frames = pitch.get_number_of_frames()
    voice = pitch.count_voiced_frames()
    voice_pct = 100 - (voice/total_frames)*100

    df = pd.DataFrame([voice_pct], columns=[measures['silence_ratio']])
    return df

def read_audio(path):
    """
    ------------------------------------------------------------------------------------------------------

    Reads an audio file and returns the Praat sound object and a dictionary of measures names.

    Parameters:
    ...........
    path : str
        The path to the audio file.

    Returns:
    ...........
    sound : praat sound object
        the Praat sound object for the given audio file.
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    ------------------------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, '../config/acoustic.json'))

    file = open(measure_path)
    measures = json.load(file)
    sound = Sound(path)

    return sound, measures

def pitchfreq(sound, measures, f0min, f0max):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the fundamental frequency values for each frame in the given audio file.

    Parameters:
    ...........
    sound : sound object
        a praat sound object
    measures : dict
        a dictionary containing the measures names for the calculated statistics.
    f0min : int
        the minimum pitch frequency value.
    f0max : int
        the maximum pitch frequency value.

    Returns:
    ...........
    df_pitch : pandas dataframe
        A dataframe containing the fundamental frequency values for each frame in the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    freq = pitch.selected_array['frequency']

    df_pitch = pd.DataFrame(list(freq), columns= [measures['fundfreq']])
    return df_pitch

def formfreq(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the formant frequency of an audio file.

    Parameters:
    ...........
    sound : sound obj
        a Praat sound object
    measures : config obj
        measures config object

    Returns:
    ...........
    df_formant : pandas dataframe
        a dataframe containing formant frequency values

    ------------------------------------------------------------------------------------------------------
    """
    formant_dict = {}
    formant = sound.to_formant_burg(time_step=.01)

    for i in range(4):
        formant_values = call(formant, "To Matrix", i+1).values[0,:]
        formant_dict['form' + str(i+1) + 'freq'] = list(formant_values)

    cols = [measures['form1freq'], measures['form2freq'], measures['form3freq'], measures['form4freq']]
    df_formant = pd.DataFrame(formant_dict)
    
    df_formant.columns = cols
    return df_formant

def loudness(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the audio intensity of an audio file.

    Parameters:
    ...........
    sound : sound obj
        a Praat sound object
    measures : config obj
        measures config object

    Returns:
    ...........
    df_loudness : dataframe;
        dataframe containing audio intensity values

    ------------------------------------------------------------------------------------------------------
    """
    intensity = sound.to_intensity(time_step=.01)
    df_loudness = pd.DataFrame(list(intensity.values[0]), columns= [measures['loudness']])
    return df_loudness

def jitter(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the jitter of an audio file.

    Parameters:
    ...........
    sound : sound obj
        A Praat sound object
    measures : config obj
        measures config object

    Returns:
    ...........
    df_jitter : pandas dataframe
        A dataframe containing jitter values

    ------------------------------------------------------------------------------------------------------
    """
    pulse = call(sound, "To PointProcess (periodic, cc)...", 75, 500)
    localJitter = call(pulse, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    localabsJitter = call(pulse, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

    rapJitter = call(pulse, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pulse, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pulse, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    cols = [measures['jitter'], measures['jitterabs'], measures['jitterrap'], measures['jitterppq5'],
           measures['jitterddp']]
    vals = [localJitter, localabsJitter, rapJitter, ppq5Jitter, ddpJitter]

    df_jitter = pd.DataFrame([vals], columns= cols)
    return df_jitter

def shimmer(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the shimmer of an audio file.

    Parameters:
    ...........
    sound : sound obj
        a Praat sound object
    measures : obj
        measures config object

    Returns:
    ...........
    df_shimmer : pandas dataframe
        dataframe containing shimmer values

    ------------------------------------------------------------------------------------------------------
    """
    pulse = call(sound, "To PointProcess (periodic, cc)...", 80, 500)
    localshimmer = call([sound, pulse], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pulse], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    apq3Shimmer = call([sound, pulse], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pulse], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pulse], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pulse], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    cols = [measures['shimmer'], measures['shimmerdb'], measures['shimmerapq3'], measures['shimmerapq5'],
           measures['shimmerapq11'], measures['shimmerdda']]
    vals = [localshimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer]

    df_shimmer = pd.DataFrame([vals], columns= cols)
    return df_shimmer

def harmonic_ratio(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the harmonic noise ratio of an audio file.

    Parameters:
    ...........
    sound : sound obj
        Praat sound object
    measures : config obj
        measures config object

    Returns
    ...........
    df_hnr : dataframe
        dataframe containing harmonic noise ratio values


    ------------------------------------------------------------------------------------------------------
    """
    hnr = sound.to_harmonicity_cc(time_step=.01)
    hnr_values = hnr.values[0]

    hnr_values = np.where(hnr_values==-200, np.NaN, hnr_values)
    df_hnr = pd.DataFrame(hnr_values, columns= [measures['hnratio']])
    return df_hnr

def glottal_ratio(sound, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the glottal noise ratio of an audio file.

    Parameters:
    ...........
    sound : sound obj
        a Praat sound object
    measures : config obj
        measures config object

    Returns:
    ...........
    df_gne : pandas dataframe
        dataframe containing glottal noise ratio values

    ------------------------------------------------------------------------------------------------------
    """
    gne = sound.to_harmonicity_gne()
    gne_values = gne.values

    gne_values = np.where(gne_values==-200, np.NaN, gne_values)
    gne_max = np.nanmax(gne_values)

    df_gne = pd.DataFrame([gne_max], columns= [measures['gneratio']])
    return df_gne

def get_voice_silence(sound, min_silence, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts the silence window of an audio file.

    Parameters:
    ...........
    sound : str
        path to the audio file
    min_silence : int
        minimum silence window length
    measures : obj
        measures config object

    Returns:
    ...........
    df_silence : pandas dataframe
        A dataframe containing the silence window values

    ------------------------------------------------------------------------------------------------------
    """
    if sound.endswith('.wav'):
        audio = AudioSegment.from_wav(sound)
    elif sound.endswith('.mp3'):
        audio = AudioSegment.from_mp3(sound)
    else:
        raise ValueError("Unsupported file format")

    dBFS = audio.dBFS

    thresh = dBFS-16
    slnc_val = silence.detect_silence(audio, min_silence_len = min_silence, silence_thresh = thresh)
    slnc_interval = [((start/1000),(stop/1000)) for start,stop in slnc_val] #in sec

    cols = [measures['silence_start'], measures['silence_end']]
    df_silence = pd.DataFrame(slnc_interval, columns=cols)

    df_silence[measures['voicesilence']] = df_silence[cols[1]] - df_silence[cols[0]]
    return df_silence

def cpp(x, fs, normOpt, dBScaleOpt): 
    """
    ------------------------------------------------------------------------------------------------------
    Computes cepstral peak prominence for a given signal 

    Parameters
    ...........
    x: ndarray
        The audio signal
    fs: integer
        The sampling frequency
    normOpt: string
        'line', 'mean' or 'nonorm' for selecting normalisation type
    dBScaleOpt: binary
        True or False for using decibel scale

    Returns
    ...........
    cpp: ndarray
        The CPP with time values 
    """
    # Settings
    frame_length = int(np.round_(0.04*fs))
    frame_shift = int(np.round_(0.01*fs))
    half_len = int(np.round_(frame_length/2))
    x_len = len(x)
    frame_len = half_len*2 + 1
    NFFT = 2**(math.ceil(np.log(frame_len)/np.log(2)))
    quef = np.linspace(0, frame_len/1000, NFFT)  
    
    # Allowed quefrency range
    pitch_range=[60, 333.3]
    quef_lim = [int(np.round_(fs/pitch_range[1])), int(np.round_(fs/pitch_range[0]))]
    quef_seq = range(quef_lim[0]-1, quef_lim[1])
    
    # Time samples
    time_samples = np.array(range(frame_length+1, x_len-frame_length+1, frame_shift))
    N = len(time_samples)
    frame_start = time_samples-half_len
    frame_stop = time_samples+half_len
    
    # High-pass filtering
    HPfilt_b = [1 - 0.97]
    x = scipy.signal.lfilter( HPfilt_b, 1, x )
    
    # Frame matrix
    frameMat = np.zeros([NFFT, N])
    for n in range(0, N):
        frameMat[0: frame_len, n] = x[frame_start[n]-1:frame_stop[n]]
        
    # Hanning
    def hanning(N):
        x = np.array([i/(N+1) for i in range(1,int(np.ceil(N/2))+1)])
        w = 0.5-0.5*np.cos(2*np.pi*x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int((np.ceil(N%2))):]))
    win = hanning(frame_len)
    winmat = np.matlib.repmat(win, N, 1).transpose()
    frameMat = frameMat[0:frame_len, :]*winmat
    
    # Cepstrum
    SpecMat = np.abs(np.fft.fft(frameMat, axis=0))
    SpecdB = 20*np.log10(SpecMat)
    if dBScaleOpt:
        ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB, axis=0)))
    else:
        ceps = 2*np.log(np.abs(np.fft.fft(SpecdB, axis=0)))

    # Finding the peak
    ceps_lim = ceps[quef_seq, :]
    ceps_max = ceps_lim.max(axis=0)
    max_index = ceps_lim.argmax(axis=0)
    
    # Normalisation
    ceps_norm = np.zeros([N])
    if normOpt=='line':
        for n in range(0, N):
            p = np.polyfit(quef_seq, ceps_lim[:,n],1)
            ceps_norm[n] = np.polyval(p, quef_seq[max_index[n]])
    elif normOpt == 'mean':
        ceps_norm = np.mean(ceps_lim)

    cpp = ceps_max-ceps_norm
    
    return cpp

def get_cepstral_features(audio_path, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts cepstral features from an audio file.

    Parameters:
    ...........
    audio_path : str
        the path to the audio file.
    measures : obj
        measures config object

    Returns:
    ...........
    df_cepstral : pandas dataframe
        a dataframe containing the cepstral features of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)

    mfccs_mean = np.mean(mfccs.T, axis=0).tolist()
    mfccs_var = np.var(mfccs.T, axis=0).tolist()

    cpp_0 = cpp(y, sr, 'line', True)
    cpp_0_mean = np.mean(cpp_0)
    cpp_0_var = np.var(cpp_0)

    df_cepstral = pd.DataFrame([mfccs_mean + mfccs_var + [cpp_0_mean] + [cpp_0_var]], columns = [
        measures['mfcc1_mean'], measures['mfcc2_mean'], measures['mfcc3_mean'], measures['mfcc4_mean'],
        measures['mfcc5_mean'], measures['mfcc6_mean'], measures['mfcc7_mean'], measures['mfcc8_mean'],
        measures['mfcc9_mean'], measures['mfcc10_mean'], measures['mfcc11_mean'], measures['mfcc12_mean'],
        measures['mfcc13_mean'], measures['mfcc14_mean'], measures['mfcc1_var'], measures['mfcc2_var'],
        measures['mfcc3_var'], measures['mfcc4_var'], measures['mfcc5_var'], measures['mfcc6_var'],
        measures['mfcc7_var'], measures['mfcc8_var'], measures['mfcc9_var'], measures['mfcc10_var'],
        measures['mfcc11_var'], measures['mfcc12_var'], measures['mfcc13_var'], measures['mfcc14_var'],
        measures['cpp_mean'], measures['cpp_var']
    ])
    return df_cepstral

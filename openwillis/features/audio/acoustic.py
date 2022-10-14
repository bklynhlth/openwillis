# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import numpy as np
import pandas as pd

import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment,silence

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def vocal_acoustics(audio_path):
    """
    -----------------------------------------------------------------------------------------
    
    Vocal acoustic variables
    
    Args:
        path: image path 
        
    Returns:
        results: Acoustic features
        
    -----------------------------------------------------------------------------------------
    """
    sound, measures = read_audio(audio_path)
    df_pitch = pitchfreq(sound, measures, 75, 500)
    df_loudness = loudness(sound, measures)
    
    df_jitter = jitter(sound, measures)
    df_shimmer = shimmer(sound, measures)
    
    df_hnr = harmonic_ratio(sound, measures)
    df_gne = glottal_ratio(sound, measures)
    df_formant = formfreq(sound, measures)
    df_silence = get_voice_silence(audio_path, 500, measures)
    
    framewise = pd.concat([df_pitch, df_loudness, df_hnr, df_formant], axis=1)
    sig_df = pd.concat([df_jitter, df_shimmer, df_gne], axis=1)
    
    df_summary = get_summary(sound, framewise, sig_df, df_silence, measures)
    return framewise, df_silence, df_summary

def common_summary(df, col_name):
    mean = df.mean()
    std = df.std()
    min_val = df.min()
    max_val = df.max()
    range_val = max_val - min_val
    
    values = [mean, std, min_val, max_val, range_val]
    cols = [col_name + '_mean', col_name + '_stddev', col_name + '_min', col_name + '_max', 
            col_name + '_range']
    
    df_summ = pd.DataFrame([values], columns= cols)
    return df_summ

def silence_summary(sound, df, measure):
    duration = call(sound, "Get total duration")
    
    df_silence = df[measure['voicesilence']]
    mean = df_silence.mean()
    
    num_pause = (len(df_silence)/duration)*60
    cols = [measure['pause_meandur'], measure['pause_rate']]
    
    silence_summ = pd.DataFrame([[mean, num_pause]], columns = cols)
    return silence_summ

def get_summary(sound, framewise, sig_df, df_silence, measure):
    df_list = []
    col_list = list(framewise.columns)
    
    for col in col_list:
        com_summ = common_summary(framewise[col], col)
        df_list.append(com_summ)
    
    summ_silence = silence_summary(sound, df_silence, measure)
    voice_pct = voice_frame(sound, measure)
    
    df_concat = pd.concat(df_list+ [sig_df, summ_silence, voice_pct], axis=1)
    return df_concat

def voice_frame(sound, measure):
    pitch = call(sound, "To Pitch", 0.0, 75, 500)
    
    total_frames = pitch.get_number_of_frames()
    voice = pitch.count_voiced_frames()
    voice_pct = 100 - (voice/total_frames)*100
    
    df = pd.DataFrame([voice_pct], columns=[measure['silence_ratio']])
    return df

def read_audio(path):
    """
    -----------------------------------------------------------------------------------------
    
    Reading audio file
    
    Args:
        path: image path 
        
    Returns:
        sound: sound object ; measures: json object
        
    -----------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/acoustic.json'))
    
    file = open(measure_path)
    measures = json.load(file)
    sound = parselmouth.Sound(path)
    
    return sound, measures

def pitchfreq(sound, measures, f0min, f0max):
    """
    -----------------------------------------------------------------------------------------
    
    Audio fundamental freq
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: Pitch frequency dataframe
        
    -----------------------------------------------------------------------------------------
    """
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    freq = pitch.selected_array['frequency']
    
    df_pitch = pd.DataFrame(list(freq), columns= [measures['fundfreq']])
    return df_pitch

def formfreq(sound, measures):
    """
    -----------------------------------------------------------------------------------------
    
    Formant freq
    
    Args:
        path: image path 
        
    Returns:
        results: facemesh object
        
    -----------------------------------------------------------------------------------------
    """
    formant_dict = {}
    formant = sound.to_formant_burg(time_step=.01)
    
    for i in range(4):
        formant_values = call(formant, "To Matrix", i+1).values[0,:]
        formant_dict['form' + str(i+1) + 'freq'] = list(formant_values)
    
    cols = [measures['form1freq'], measures['form2freq'], measures['form3freq'], measures['form4freq']]
    df_formant = pd.DataFrame(formant_dict)
    return df_formant

def loudness(sound, measures):
    """
    -----------------------------------------------------------------------------------------
    
    Audio intensity
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: Audio intensity dataframe
        
    -----------------------------------------------------------------------------------------
    """
    intensity = sound.to_intensity(time_step=.01)
    df_loudness = pd.DataFrame(list(intensity.values[0]), columns= [measures['loudness']])
    return df_loudness

def jitter(sound, measures):
    """
    -----------------------------------------------------------------------------------------
    
    Jitter
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: Jitter dataframe
        
    -----------------------------------------------------------------------------------------
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
    -----------------------------------------------------------------------------------------
    
    Shimmer
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: Shimmer dataframe
        
    -----------------------------------------------------------------------------------------
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
    -----------------------------------------------------------------------------------------
    
    Harmonic Noise Ratio
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: HNR dataframe
        
    -----------------------------------------------------------------------------------------
    """
    hnr = sound.to_harmonicity_cc(time_step=.01)
    hnr_values = hnr.values[0]
    
    hnr_values = np.where(hnr_values==-200, np.NaN, hnr_values)
    df_hnr = pd.DataFrame(hnr_values, columns= [measures['hnratio']])
    return df_hnr

def glottal_ratio(sound, measures):
    """
    -----------------------------------------------------------------------------------------
    
    Glottal Noise Ratio
    
    Args:
        sound: Praat sound object
        measures: measures config object
        
    Returns:
        results: GNE dataframe
        
    -----------------------------------------------------------------------------------------
    """
    gne = sound.to_harmonicity_gne()
    gne_values = gne.values
    
    gne_values = np.where(gne_values==-200, np.NaN, gne_values)
    gne_max = np.nanmax(gne_values)
    
    df_gne = pd.DataFrame([gne_max], columns= [measures['gneratio']])
    return df_gne

def get_voice_silence(sound, min_silence, measures):
    """
    -----------------------------------------------------------------------------------------
    
    Get silence window
    
    Args:
        sound: Audio file path
        min_silence: Minimum silence window length
        measures: measures config object
        
    Returns:
        results: silence window dataframe
        
    -----------------------------------------------------------------------------------------
    """
    audio = AudioSegment.from_wav(sound)
    dBFS = audio.dBFS
    
    thresh = dBFS-16
    slnc_val = silence.detect_silence(audio, min_silence_len = min_silence, silence_thresh = thresh)
    slnc_interval = [((start/1000),(stop/1000)) for start,stop in slnc_val] #in sec
    
    cols = [measures['silence_start'], measures['silence_end']]
    df_silence = pd.DataFrame(slnc_interval, columns=cols)
    
    df_silence[measures['voicesilence']] = df_silence[cols[1]] - df_silence[cols[0]]
    return df_silence
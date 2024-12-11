# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com
import logging

import numpy as np
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def resample(audio_signal, sample_rate):
    """
    ------------------------------------------------------------------------------------------------------

    Resamples the audio signal to the target sample rate.

    Parameters:
    ...........
    audio_signal : pydub.AudioSegment
        input audio signal
    sample_rate : int
        target sample rate

    Returns:
    ...........
    pydub.AudioSegment
        resampled audio signal

    ------------------------------------------------------------------------------------------------------
    """

    audio_signal = audio_signal.set_frame_rate(sample_rate)
    return audio_signal

def dc_offset(audio_signal):
    """
    ------------------------------------------------------------------------------------------------------

    Removes the DC offset from the audio signal.

    Parameters:
    ...........
    audio_signal : pydub.AudioSegment
        input audio signal

    Returns:
    ...........
    pydub.AudioSegment
        audio signal with DC offset removed

    ------------------------------------------------------------------------------------------------------
    """

    num_channels = audio_signal.channels

    if num_channels == 1:
        offset = audio_signal.get_dc_offset(channel=1)
        if offset == 0:
            return audio_signal
        audio_signal.remove_dc_offset(channel=1, offset=offset)
    else:
        offset_l = audio_signal.get_dc_offset(channel=1)
        offset_r = audio_signal.get_dc_offset(channel=2)
        offset = (offset_l + offset_r) / 2

        if offset == 0:
            return audio_signal

        audio_signal.remove_dc_offset(channel=1, offset=offset_l)
        audio_signal.remove_dc_offset(channel=2, offset=offset_r)

    return audio_signal

def volume_normalization(audio_signal, target_dBFS):
    """
    ------------------------------------------------------------------------------------------------------
    
    Normalizes the volume of the audio signal to the target dBFS.
    
    Parameters:
    ...........
    audio_signal : pydub.AudioSegment
        input audio signal
    target_dBFS : float
        target dBFS
        
    Returns:
    ...........
    pydub.AudioSegment
        normalized audio signal

    ------------------------------------------------------------------------------------------------------
    """

    headroom = -audio_signal.max_dBFS
    gain_adjustment = target_dBFS - audio_signal.dBFS

    if gain_adjustment > headroom:
        gain_adjustment = headroom

    audio_signal = audio_signal.apply_gain(gain_adjustment)
    return audio_signal

def audio_preprocess(audio_in):
    """
    ------------------------------------------------------------------------------------------------------

    Preprocesses audio signal by resampling, removing DC offset, normalizing volume, and denoising.

    Parameters:
    ...........
    audio_in : str
        path to the input audio file

    Returns:
    ...........
    signal_dict: dict
        dictionary containing the preprocessed audio signal

    ------------------------------------------------------------------------------------------------------
    """


    signal_dict = {}

    try:
        if not audio_in.endswith(".wav") and not audio_in.endswith(".mp3"):
            logger.info(f'Error in audio preprocessing- file: {audio_in} & Error: File format not supported')
            return

        audio_signal = AudioSegment.from_file(audio_in, format="wav" if audio_in.endswith(".wav") else "mp3")

        audio_signal = resample(audio_signal, 44100)
        audio_signal = dc_offset(audio_signal)
        audio_signal = volume_normalization(audio_signal, -20)

        signal_dict = {'clean': np.array(audio_signal.get_array_of_samples())}

    except Exception as e:
        logger.info(f'Error in audio preprocessing- file: {audio_in} & Error: {e}')
    finally:
        return signal_dict

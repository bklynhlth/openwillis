import pandas as pd
import numpy as np

def transcribe_response_to_dataframe(response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(cloud:AWS) a response from a speech-to-text service into a pandas DataFrame.

    Parameters:
    ----------
    response : dict
        The response object containing the transcribed data.

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.
    speakers: int
        The number of speakers detected in the transcription.

    ------------------------------------------------------------------------------------------------------
    """
    speakers = 0
    df = pd.DataFrame()

    if 'results' in response:
        if 'speaker_labels' in response['results']:

            if 'speakers' in response['results']['speaker_labels']:
                speakers = response['results']['speaker_labels']["speakers"]

            if 'items' in response['results']:
                items = response['results']["items"]
                df = pd.DataFrame(items)

                df["confidence"] = df["alternatives"].apply(lambda x: x[0]["confidence"])
                df["content"] = df["alternatives"].apply(lambda x: x[0]["content"])
                df["confidence"] = df["confidence"].astype(float)

                df = df[df["confidence"] > 0].reset_index(drop=True)
                df = df[["start_time", "end_time", "confidence", "speaker_label", "content"]]
                
    return df, speakers

def extract_data(segment_info):
    """
    ------------------------------------------------------------------------------------------------------

    extract data from word_info

    Parameters:
    ----------
    segment_info : object
        The phrase level transcribed data.

    Returns:
    -------
    df : pandas series
        The phrase level transcribed data in a pandas series.

    ------------------------------------------------------------------------------------------------------
    """
    words = segment_info.get("words", None)

    starts = [word.get("start", np.nan) for word in words]
    ends = [word.get("end", np.nan) for word in words]
    phrases = [word.get("word", "") for word in words]
    scores = [word.get("score", 0) for word in words]
    speakers = [segment_info.get("speaker", "no_speaker") for _ in words]

    return pd.DataFrame({"start": starts, "end": ends, "phrase": phrases, "score": scores, "speaker": speakers})

def whisperx_to_dataframe(json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(local:whisperx) a json response into a pandas DataFrame.

    Parameters:
    ----------
    json_response : dict
        The response object containing the transcribed data.

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.
    speakers: int
        The number of speakers detected in the transcription.

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(columns=["start_time", "end_time", "content", "confidence", "speaker_label"])
    if 'segments' in json_response:
        
        segment_infos = json_response["segments"]
        df = pd.DataFrame(segment_infos).apply(extract_data, axis=1)
        df = pd.concat(df.tolist(), ignore_index=True)

        df = df[df["score"] > 0].reset_index(drop=True)
        df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
        
        df = df[df["speaker"] != "no_speaker"].reset_index(drop=True)
        df = df.rename(columns={"start": "start_time", "end": "end_time", "score": "confidence", "speaker": "speaker_label", "phrase": "content"})

    speakers = df['speaker_label'].nunique()
    return df, speakers

def vosk_to_dataframe(json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(local:vosk) a json response into a pandas DataFrame.

    Parameters:
    ----------
    json_response : dict
        The response object

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(columns=["start_time", "end_time", "content", "confidence", "speaker_label"])
        
    df = pd.DataFrame(json_response)

    df = df[df["conf"] > 0].reset_index(drop=True)
    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    
    df = df.rename(columns={"start": "start_time", "end": "end_time", "conf": "confidence", "word": "content"})
    df['speaker_label'] = 'speaker0'

    return df

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

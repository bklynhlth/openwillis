# website:   http://www.brooklyn.health

# import the required packages
import os
import wave
import json
import logging
import json

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from openwillis.measures.audio.util import util as ut

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def get_config():
    """
    ------------------------------------------------------------------------------------------------------

    Load the configuration settings for the speech transcription.

    Parameters:
    ...........
    None

    Returns:
    ...........
    measures : dict
        A dictionary containing the configuration settings.

    ------------------------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def filter_audio(filepath, t_interval):
    """
    ------------------------------------------------------------------------------------------------------

    Filter an audio file to extract a segment based on the specified time interval.

    Parameters:
    ............
    filepath : str
        The path to the audio file to be filtered.
    t_interval : list
        A list of tuples representing the start and end times (in seconds) of the segment to extract.

    Returns:
    ............
    sound : AudioSegment
        The filtered audio segment.

    ------------------------------------------------------------------------------------------------------
    """
    sound = AudioSegment.from_wav(filepath)

    if len(t_interval)==2:
        sound = sound[int(t_interval[0])*1000 : int(t_interval[1])*1000]

    elif len(t_interval)==1:
        sound = sound[int(t_interval[0])*1000:]

    sound = sound.set_channels(1)
    return sound

def filter_speech(measures, results):
    """
    ------------------------------------------------------------------------------------------------------

    Filter the speech transcription results to extract the transcript.

    Parameters:
    ...........
    measures : dict
        A dictionary containing the configuration settings for the speech transcription.
    results : list of dict
        The raw transcription results returned by the transcription service.

    Returns:
    ...........
    result_key : list
        A list containing the framewise transcription of the audio file.
    transcript : str
        The transcript of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    result_key = []
    text_key = []
    transcript_dict = {}

    for res in results:
        dict_keys = res.keys()

        if 'result' in dict_keys and 'text' in dict_keys:
            result_key.extend(res['result'])
            text_key.append(res['text'])

    transcript_dict['result'] = result_key
    transcript_dict['text'] = ' '.join(text_key)
    return result_key, ' '.join(text_key)

def get_vosk(audio_path, lang):
    """
    ------------------------------------------------------------------------------------------------------

    Recognize speech using the Vosk model.

    Parameters:
    ............
    audio_path : str
        The path to the audio file to be transcribed.
    lang : str
        The language of the audio file (e.g. 'en-us', 'es', 'fr').

    Returns:
    ............
    results : list of dict
        The raw transcription results returned by the Vosk model.

    ------------------------------------------------------------------------------------------------------
    """
    model = Model(lang=lang)
    wf = wave.open(audio_path, "rb")

    recog = KaldiRecognizer(model, wf.getframerate())
    recog.SetWords(True)

    results = []
    while True:

        data = wf.readframes(4000) #Future work
        if len(data) == 0:
            break

        if recog.AcceptWaveform(data):
            partial_result = json.loads(recog.Result())
            results.append(partial_result)

    partial_result = json.loads(recog.FinalResult())
    results.append(partial_result)
    return results

def stereo_to_mono(filepath, t_interval):
    """
    ------------------------------------------------------------------------------------------------------

    Convert a stereo audio file to a mono audio file.

    Parameters:
    ............
    filepath : str
        The path to the stereo audio file to be converted.
    t_interval : list
        A list of tuples representing the start and end times (in seconds) of segments of the audio file to be transcribed.

    Returns:
    ............
    mono_filepath : str
        The path to the mono audio file.

    ------------------------------------------------------------------------------------------------------
    """
    sound = filter_audio(filepath, t_interval)

    filename, _ = os.path.splitext(os.path.basename(filepath))
    dir_name = os.path.join(os.path.dirname(filepath), 'temp_mono_' + filename)

    ut.make_dir(dir_name)
    mono_filepath = os.path.join(dir_name, filename + '.wav')
    sound.export(mono_filepath, format="wav")
    return mono_filepath

def run_vosk(filepath, language, transcribe_interval = []):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe speech in an audio file using the Vosk model.

    Parameters:
    ............
    filepath : str
        The path to the audio file to be transcribed.
    language : str, optional
        The language of the audio file (e.g. 'en-us', 'es', 'fr'). Default is 'en-us'.
    transcribe_interval : list, optional
        A list of tuples representing the start and end times (in seconds) of segments of the audio file to be transcribed.
        Default is an empty list.

    Returns:
    ............
    json_response : str
        The JSON response from the Vosk transcription service.
    transcript : str
        The transcript of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    json_response = json.dumps({})
    transcript = mono_filepath = ''

    try:
        if os.path.exists(filepath):

            measures = get_config()
            mono_filepath = stereo_to_mono(filepath, transcribe_interval)
            results = get_vosk(mono_filepath, language)

            ut.remove_dir(os.path.dirname(mono_filepath)) #Clean temp directory
            json_response, transcript = filter_speech(measures, results)

        else:
            logger.info(f'Audio file not available. File: {filepath}')

    except Exception as e:
        ut.remove_dir(os.path.dirname(mono_filepath))#Clean temp directory
        logger.error(f'Error in speech Transcription: {e} & File: {filepath}')

    finally:
        return json_response, transcript

    

def speech_transcription_vosk(filepath, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------

    Speech transcription function that transcribes an audio file using vosk.

    Parameters:
    ...........
    filepath : str
        The path to the audio file to be transcribed.
    language : str, optional
        The language of the audio file (e.g. 'en-us', 'es', 'fr'). Default is 'en-us'.
    transcribe_interval : list, optional
        A list of tuples representing the start and end times (in seconds) of segments of the audio file to be transcribed.
        Only applicable if model is 'vosk'. Default is an empty list.

    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format
    transcript : str
        The transcript of the recording.

    ------------------------------------------------------------------------------------------------------
    """

    measures = get_config()
    language = kwargs.get('language', 'en-us')
    transcribe_interval = kwargs.get('transcribe_interval', [])
    
    json_response, transcript = run_vosk(filepath, language, transcribe_interval)
    return json_response, transcript

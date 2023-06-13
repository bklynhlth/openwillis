# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
from pydub import AudioSegment

def make_dir(dir_name):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a directory if it doesn't already exist.

    Parameters:
    ...........
    dir_name : str
        The path to the directory

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def to_audio(filepath, speaker_label, out_dir):
    """
    ------------------------------------------------------------------------------------------------------

    Save a numpy signal into multiple speaker-specific audio files.

    Parameters:
    ----------
        filepath : str
            The path to the input audio file.
        speaker_label : dict
            A dictionary containing speaker labels as keys and corresponding segments (NumPy arrays) as values.
        out_dir : str
            The directory where the output audio files will be saved.

    ------------------------------------------------------------------------------------------------------
    """
    make_dir(out_dir)
    for key, value in speaker_label.items():
        file_name, _ = os.path.splitext(os.path.basename(filepath))

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        spk_signal = AudioSegment(value.tobytes(), frame_rate=audio_signal.frame_rate,
                                  sample_width=audio_signal.sample_width, channels=audio_signal.channels)

        output_file = os.path.join(out_dir, file_name + '_' + key + '.wav')
        spk_signal.export(output_file, format="wav")
# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com
import logging
import os
import re
import shlex
import subprocess
import tempfile

from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def resample(audio_signal, sample_rate):

    audio_signal = audio_signal.set_frame_rate(sample_rate)
    return audio_signal

def dc_offset(audio_signal):

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

    audio_signal = audio_signal.apply_gain(target_dBFS - audio_signal.dBFS)
    return audio_signal

def find_silence(audio_signal, silence_threshold, silence_duration):

    # save audio signal to a temporary file
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    silence_audio = None

    try:
        audio_signal.export(audio_file.name, format="wav")

        args = shlex.split(f'ffmpeg -i "{audio_file.name}" -af silencedetect=n={silence_threshold}dB:d={silence_duration} -f null -')
        process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        sil_list = []

        if process.returncode == 0:
            ffmpeg_output = process.stderr.decode("utf-8")

            silence_start, silence_end, silence_duration = None, None, None

            for line in ffmpeg_output.split("\n"):
                line = line.strip()
                if "silence_start" in line:
                    silence_start = str(re.search("silence_start: (.+)", line).group(1))
                if "silence_end" in line:
                    silence_end = str(
                        re.search("silence_end: ([-]?[0-9]*\.?[0-9]+)", line).group(1)
                    )
                if "silence_duration" in line:
                    silence_duration = str(
                        re.search(
                            "silence_duration: ([-]?[0-9]*\.?[0-9]+)", line
                        ).group(1)
                    )
                    if silence_start is not None and silence_end is not None:
                        sil_list.append([silence_start, silence_end, silence_duration])

        # get the silence with the longest duration
        if sil_list:
            longest_silence = max(sil_list, key=lambda x: float(x[2]))
            start = float(longest_silence[0])
            end = float(longest_silence[1])

            silence_audio = audio_signal[start * 1000 : end * 1000]
        else:
            silence_audio = None
    except Exception as e:
        logger.error(f'find_silence failed: {e}')
    finally:
        audio_file.close()
        os.unlink(audio_file.name)

    return silence_audio

def remove_background_noise(audio_signal, audio_silence, sensitivity):

    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_silence_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    noise_profile_file = tempfile.NamedTemporaryFile(delete=False, suffix=".prof")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    try:
        audio_signal.export(audio_file.name, format="wav")
        audio_silence.export(audio_silence_file.name, format="wav")

        # Generate noise profile from the silence audio file
        noise_profile_command = shlex.split(
            f'sox "{audio_silence_file.name}" -n noiseprof "{noise_profile_file.name}"'
        )
        noise_profile_process = subprocess.run(noise_profile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Apply noise reduction if noise profile generation was successful
        if noise_profile_process.returncode == 0:
            noise_reduction_command = shlex.split(
                f'sox "{audio_file.name}" "{output_file.name}" noisered "{noise_profile_file.name}" {sensitivity}'
            )
            noise_reduction_process = subprocess.run(noise_reduction_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Load the processed audio if noise reduction was successful
            if noise_reduction_process.returncode == 0:
                audio_signal = AudioSegment.from_file(output_file.name)

    except Exception as e:
        logger.error(f'remove_background_noise failed: {e}')
    finally:
        audio_file.close()
        audio_silence_file.close()
        noise_profile_file.close()
        output_file.close()
        os.unlink(audio_file.name)
        os.unlink(audio_silence_file.name)
        os.unlink(noise_profile_file.name)
        os.unlink(output_file.name)

    return audio_signal

def denoise(audio_signal):

    audio_silence = find_silence(audio_signal, -30, 0.5)
    if audio_silence is None:
        return audio_signal
    
    audio_signal = remove_background_noise(audio_signal, audio_silence, 0.21)

    return audio_signal

def audio_preprocess(audio_in, audio_out):

    try:
        if not audio_in.endswith(".wav") and not audio_in.endswith(".mp3"):
            logger.error(f'Error in audio preprocessing- file: {audio_in} & Error: File format not supported')
            return
        
        if not os.path.exists(audio_in):
            logger.error(f'Error in audio preprocessing- file: {audio_in} & Error: File not found')
            return

        audio_signal = AudioSegment.from_file(audio_in, format="wav" if audio_in.endswith(".wav") else "mp3")

        audio_signal = resample(audio_signal, 16000)
        audio_signal = dc_offset(audio_signal)
        audio_signal = volume_normalization(audio_signal, -20)
        audio_signal = denoise(audio_signal)

        audio_signal.export(audio_out, format="wav" if audio_out.endswith(".wav") else "mp3")

    except Exception as e:
        logger.error(f'Error in audio preprocessing- file: {audio_in} & Error: {e}')



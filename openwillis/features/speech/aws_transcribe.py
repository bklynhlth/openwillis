# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import boto3
import urllib

import json
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def transcribe_audio(s3uri, region_name, job_name, language = 'en-US'):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe an audio file using AWS Transcribe.

    Parameters:
    ...........
    s3uri : str
        The S3 URI of the input audio file.
    region_name : str
        The region where the transcription should be done.
    job_name : str
        The name of the transcription job.
    language : str, optional
        The language used in the input audio file. Default is 'en-US'.

    Raises:
    ...........
    Exception: If the transcription job fails.

    Returns:
    ...........
    tuple : Two strings, the first string is the response object in JSON format, and the second string is
    the transcript of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    response = "{}"
    transcript = ""

    try:
        transcribe = boto3.client('transcribe', region_name = region_name)
        settings = {'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 2}

        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3uri},

            #IdentifyMultipleLanguages=True,
            LanguageCode=language,
            Settings=settings
        )

        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            #logger.info("Not ready yet...")

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            read_data = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            response = json.loads(read_data.read().decode('utf-8'))

    except Exception as e:
        logger.error(f"Transcription job failed with file: {s3uri} exception: {e}")

    finally:
        return response, transcript

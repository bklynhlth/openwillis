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
    Transcribe an audio file using AWS Transcribe.

    Args:
        job_uri (str): The S3 URI of the input audio file.
        output_bucket (str): The name of the S3 bucket where the output file should be saved.

    Raises:
        Exception: If the transcription job fails.

    Returns:
        str: The transcript of the audio file.
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
            logger.info("Not ready yet...")

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            read_data = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            response = json.loads(read_data.read().decode('utf-8'))

    except Exception as e:
        logger.error("Transcription job failed with exception: {}".format(e))

    finally:
        return response, transcript

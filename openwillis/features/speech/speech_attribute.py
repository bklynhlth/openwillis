# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import pandas as pd

import logging
from openwillis.features.speech import util as ut

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def get_config():
    """
    -----------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes, and returns the
    contents of the file as a dictionary.

    Returns:
        measures: A dictionary containing the names of the columns in the output dataframes.

    -----------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def create_empty_dataframes(measures):
    """
    -----------------------------------------------------------------------------------------

    Creating an empty measure dataframe

    Args:
        measures: config file object

    Return:
        empty pandas dataframe

    -----------------------------------------------------------------------------------------
    """
    summ_df = pd.DataFrame(columns=[measures['tot_words'], measures['speech_verb'], measures['speech_adj'],
                                    measures['speech_pronoun'], measures['speech_noun'], measures['neg'], measures['neu'],
                                    measures['pos'], measures['compound'], measures['speech_mattr'], measures['rate_of_speech'],
                                    measures['pause_rate'], measures['pause_meandur'], measures['silence_ratio']])

    # Create an empty tag dataframe
    tag_df = pd.DataFrame(columns=[measures['word'], measures['tag']])
    return tag_df, summ_df

def is_amazon_transcribe(json_conf):
    """
    -----------------------------------------------------------------------------------------

    This function checks if the json response object is from Amazon Transcribe.

    Args:
        json_conf: JSON response object.

    Returns:
        True if the json response object is from Amazon Transcribe, False otherwise.

    -----------------------------------------------------------------------------------------
    """
    return 'jobName' in json_conf and 'results' in json_conf

def filter_transcribe(json_conf):
    """
    -----------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data for Amazon Transcribe json response objects.

    Args:
        json_conf: aws transcribe json response.

    Returns:
        text: The text extracted from the JSON object.
        filter_json: The filtered JSON object containing only the relevant data for processing.

    -----------------------------------------------------------------------------------------
    """
    text = json_conf['results']['transcripts'][0].get('transcript', '')
    item_data = json_conf['results']['items']
    filter_json = [item for item in item_data if 'start_time' in item and 'end_time' in item]
    return text, filter_json

def filter_vosk(json_conf):
    """
    -----------------------------------------------------------------------------------------

    This function extracts the text for json_conf objects from sources other than Amazon Transcribe.

    Args:
        json_conf: The input text in the form of a JSON object.

    Returns:
        text: The input text extracted from the JSON object.

    -----------------------------------------------------------------------------------------
    """
    text_list = [word['word'] for word in json_conf if 'word' in word]
    text = " ".join(text_list)
    return text

def speech_characteristics(json_conf, language='en-us'):
    """
    -----------------------------------------------------------------------------------------

    Speech Characteristics

    Args:
        json_conf: Transcribed json file
        language: Language type

    Returns:
        results: Speech Characteristics

    -----------------------------------------------------------------------------------------
    """
    measures = get_config()
    tag_df, summ_df = create_empty_dataframes(measures)

    try:
        if bool(json_conf):
            ut.download_nltk_resources()

            if is_amazon_transcribe(json_conf):
                text, filter_json = filter_transcribe(json_conf)

                if len(filter_json) > 0 and len(text) > 0:
                    tag_df, summ_df = ut.process_language_feature(filter_json, [tag_df, summ_df], text, language,
                                                               measures, ['start_time', 'end_time'])
            else:
                text = filter_vosk(json_conf)
                if len(text) > 0:
                    tag_df, summ_df = ut.process_language_feature(json_conf, [tag_df, summ_df], text, language, measures,
                                                               ['start', 'end'])

    except Exception as e:
        logger.info('Error in speech Characteristics {}'.format(e))

    finally:
        return tag_df, summ_df

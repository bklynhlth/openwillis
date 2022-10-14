# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import os
import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import nltk
from lexicalrichness import LexicalRichness
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

#NLTK Tag list
tag_dict = {'PRP': 'Pronoun', 'PRP$': 'Pronoun', 'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb' , 'VBN': 'Verb', 
            'VBP': 'Verb', 'VBZ': 'Verb', 'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective', 'NN': 'Noun', 
            'NNP': 'Noun', 'NNS': 'Noun'}

def download_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
def download_tagger():
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        
def get_tag(text, tag_dict, measures):
    tag_list = nltk.pos_tag(text.split())
    
    tag_df = pd.DataFrame(tag_list, columns=[measures['word'], measures['tag']])
    tag_df = tag_df.replace({measures['tag']: tag_dict})
    return tag_df

def get_tag_summ(tag_df, word, measures):
    word_len = len(word) if len(word)>0 else 1
    
    verb = len(tag_df[tag_df[measures['tag']] == 'Verb'])/word_len
    pronoun = len(tag_df[tag_df[measures['tag']] == 'Pronoun'])/word_len
    adj = len(tag_df[tag_df[measures['tag']] == 'Adjective'])/word_len
    noun = len(tag_df[tag_df[measures['tag']] == 'Noun'])/word_len
    
    tag_object = [len(word), verb, adj, pronoun, noun]
    cols = [measures['tot_words'], measures['speech_verb'], measures['speech_adj'], measures['speech_pronoun'], 
            measures['speech_noun']]
    summ_df = pd.DataFrame([tag_object], columns=cols)
    return summ_df

def get_sentiment(word, text, measures):
    sentiment = SentimentIntensityAnalyzer()
    
    sentiment_dict = sentiment.polarity_scores(text)
    sent_summ = pd.DataFrame([sentiment_dict.values()], columns=sentiment_dict.keys())
    
    mattr = get_mattr(word)
    sent_summ[measures['speech_mattr']] = mattr
    
    sent_summ = sent_summ.rename(columns={"pos": measures['pos'], "neg": measures['neg'], "neu": measures['neu'], 
                                          "compound": measures['compound']})
    return sent_summ

def get_mattr(word):
    filter_punc = list(value for value in word if value not in ['.','!','?'])
    filter_punc = " ".join(str(filter_punc))
    mattr = np.nan
    
    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=lex_richness.words)
    
    return mattr

def get_stats(ros, file_dur, pause_list, measures):
    pause_rate = (len(pause_list)/file_dur)*60
    
    pause_meandur = np.mean(pause_list)
    silence_ratio = np.sum(pause_list)/(file_dur - np.sum(pause_list))
    
    feature_list = [ros, pause_rate, pause_meandur, silence_ratio]
    col_list = [measures['rate_of_speech'], measures['pause_rate'], measures['pause_meandur'], 
                measures['silence_ratio']]
    
    df_stats = pd.DataFrame([feature_list], columns=col_list)
    return df_stats

def get_pause_feature(json_conf, word, measures):
    pause_list = []
    df_pause_ftr = pd.DataFrame()
    
    if len(json_conf) <=0:
        return df_pause_ftr
    
    file_dur = json_conf[-1]['end'] - json_conf[0]['start']
    ros = (len(word)/ file_dur)*60
    
    for index in range(1, len(json_conf)):
        pause_dur = json_conf[index]['start'] - json_conf[index-1]['end']
        
        if pause_dur>=0:
            pause_list.append(pause_dur)
    
    df_pfeature = get_stats(ros, file_dur, pause_list, measures)
    return df_pfeature

def get_config():
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))
    
    file = open(measure_path)
    measures = json.load(file)
    return measures

def get_langauge_feature(json_conf, text, language, measures):
    sentences = nltk.tokenize.sent_tokenize(text)
    
    tag_df = pd.DataFrame()
    word_list = nltk.tokenize.word_tokenize(text)
    pause = get_pause_feature(json_conf, word_list, measures)
    
    if language == 'en-us':
        tag_df = get_tag(text, tag_dict, measures)
        
        tag_summ = get_tag_summ(tag_df, word_list, measures)
        sent_summ = get_sentiment(word_list, text, measures)
        speech_summ = pd.concat([tag_summ, sent_summ, pause], axis=1)
    
    else:
        speech_summ = pause
    return tag_df, speech_summ
        
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
    speech_summ = pd.DataFrame()
    
    try:
        #Downloading NLTK resources
        download_punkt()
        download_tagger()
        measures = get_config()

        text_list = [word['word'] for word in json_conf if 'word' in word]
        text = " ".join(text_list)
        tag_df, speech_summ = get_langauge_feature(json_conf, text, language, measures)

    except Exception as e:
        logger.info('Error in speech Characteristics')
        
    return tag_df, speech_summ
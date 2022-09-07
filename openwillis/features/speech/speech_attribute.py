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
import librosa

from openwillis.features.speech import speech_transcribe
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
    
    tag_df = pd.DataFrame(tag_list, columns=[measures['label'], measures['tag']])
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
    mean_pause = np.mean(pause_list)
    tot_pause = np.sum(pause_list)
    
    norm_pause = tot_pause/file_dur
    num_pause = sum(i > 0 for i in pause_list)
    
    feature_list = [ros, mean_pause, norm_pause, num_pause, tot_pause]
    col_list = [measures['speech_rate'], measures['speech_pause'], measures['speech_norm_pause'], 
                measures['speech_pause_number'], measures['speech_pause_total']]
    
    df_stats = pd.DataFrame([feature_list], columns=col_list)
    return df_stats

def get_pause_feature(p_intrvl, word, audio_path, measures):
    file_dur = librosa.get_duration(filename=audio_path)
    df_pause_ftr = pd.DataFrame()
    
    if file_dur <=0:
        return df_pause_ftr
    
    pause_list = []
    ros = (len(word)/ file_dur)*60
    
    for index in range(1, len(p_intrvl)):
        pause_dur = p_intrvl[index]['start'] - p_intrvl[index-1]['end']
        
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
        
def speech_characteristic(audio_path, lang):
    
    #Downloading NLTK resources
    download_punkt()
    download_tagger()
    measures = get_config()
    
    result, text = speech_transcribe.transcribe(audio_path, lang)
    sentences = nltk.tokenize.sent_tokenize(text)
    word_list = nltk.tokenize.word_tokenize(text)
    
    tag_df = get_tag(text, tag_dict, measures)
    tag_summ = get_tag_summ(tag_df, word_list, measures)
    sent_summ = get_sentiment(word_list, text, measures)
    pause = get_pause_feature(result, word_list, audio_path, measures)
    
    speech_summ = pd.concat([tag_summ, sent_summ, pause], axis=1)
    return tag_df, speech_summ
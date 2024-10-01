# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import string
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun",
            "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb", "DT": "Determiner"}
FIRST_PERSON_PRONOUNS = ["I", "me", "my", "mine", "myself"]

def get_mattr(text, lemmatizer, window_size=50):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.
    lemmatizer : spacy lemmatizer
        The lemmatizer to be used in the calculation.
    window_size : int
        The size of the window to be used in the calculation.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """

    words = nltk.word_tokenize(text)
    words = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in words]
    words = [w for w in words if w != '']
    words_texts = [token.lemma_ for token in lemmatizer(' '.join(words))]
    filter_punc = " ".join(words_texts)
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=min(window_size, lex_richness.words))

    return mattr

def get_tag(word_df, word_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns
     word-level part-of-speech tags.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    word_list: list
        List of transcribed text at the word level.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    tag_list = nltk.pos_tag(word_list)
    
    tag_list_pos = [TAG_DICT[tag[1]] if tag[1] in TAG_DICT.keys() else "Other" for tag in tag_list]
    word_df[measures["part_of_speech"]] = tag_list_pos

    word_df[measures["first_person"]] = [word in FIRST_PERSON_PRONOUNS for word in word_list]
    # make non pronouns NaN
    word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]] == "Pronoun", np.nan)

    present_tense = ["VBP", "VBZ"]
    past_tense = ["VBD", "VBN"]
    tag_list_verb = ["Present" if tag[1] in present_tense else "Past" if tag[1] in past_tense else "Other" for tag in tag_list]
    word_df[measures["verb_tense"]] = tag_list_verb
    # make non verbs NaN
    word_df[measures["verb_tense"]] = word_df[measures["verb_tense"]].where(word_df[measures["part_of_speech"]] == "Verb", np.nan)

    return word_df

def calculate_first_person_sentiment(df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates a measure of the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    res1: list
        A list containing the calculated measure of the influence of positive sentiment on the use of first person pronouns.
    res2: list
        A list containing the calculated measure of the influence of negative sentiment on the use of first person pronouns.

    ------------------------------------------------------------------------------------------------------
    """
    
    res1 = []
    res2 = []
    for i in range(len(df)):
        perc = df.loc[i, measures["first_person_percentage"]]
        pos = df.loc[i, measures["pos"]]
        neg = df.loc[i, measures["neg"]]

        if perc is np.nan:
            res1.append(np.nan)
            res2.append(np.nan)
            continue

        res1.append((100-perc)*pos)
        res2.append(perc*neg)

    return res1, res2

def calculate_first_person_percentage(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of first person pronouns in the input text.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.

    Returns:
    ...........
    float
        The calculated percentage of first person pronouns in the input text.

    ------------------------------------------------------------------------------------------------------
    """

    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    # filter out non pronouns
    pronouns = [tag[0] for tag in tags if tag[1] == "PRP" or tag[1] == "PRP$"]
    if len(pronouns) == 0:
        return np.nan

    first_person_pronouns = len([word for word in pronouns if word in FIRST_PERSON_PRONOUNS])
    return (first_person_pronouns / len(pronouns)) * 100

def get_first_person_turn(turn_df, turn_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in each turn.
     Specifically, it calculates the percentage of first person pronouns in each turn,
     and the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    turn_list: list
        List of transcribed text at the turn level.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    first_person_percentages = [calculate_first_person_percentage(turn) for turn in turn_list]

    turn_df[measures["first_person_percentage"]] = first_person_percentages

    first_pos, first_neg = calculate_first_person_sentiment(turn_df, measures)

    turn_df[measures["first_person_sentiment_positive"]] = first_pos
    turn_df[measures["first_person_sentiment_negative"]] = first_neg

    return turn_df

def get_first_person_summ(summ_df, turn_df, full_text, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in the transcript.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    full_text: str
        The full transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures["first_person_percentage"]] = calculate_first_person_percentage(full_text)

    if len(turn_df) > 0:
        summ_df[measures["first_person_sentiment_positive"]] = turn_df[measures["first_person_sentiment_positive"]].mean(skipna=True)
        summ_df[measures["first_person_sentiment_negative"]] = turn_df[measures["first_person_sentiment_negative"]].mean(skipna=True)

        first_person_sentiment = []
        for i in range(len(turn_df)):
            if turn_df.loc[i, measures["pos"]] > turn_df.loc[i, measures["neg"]]:
                first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_positive"]])
            else:    
                first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_negative"]])

        summ_df[measures["first_person_sentiment_overall"]] = np.nanmean(first_person_sentiment)
    else:
        first_pos, first_neg = calculate_first_person_sentiment(summ_df, measures)
        summ_df[measures["first_person_sentiment_positive"]] = first_pos
        summ_df[measures["first_person_sentiment_negative"]] = first_neg

        if summ_df[measures["pos"]].values[0] > summ_df[measures["neg"]].values[0]:
            summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_positive"]].values[0]
        else:
            summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_negative"]].values[0]

    return summ_df

def get_pos_tag(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the part-of-speech measures
        and adds them to the output dataframes

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        word_list, turn_list, full_text = text_list

        word_df = get_tag(word_df, word_list, measures)

        if len(turn_list) > 0:
            turn_df = get_first_person_turn(turn_df, turn_list, measures)

        summ_df = get_first_person_summ(summ_df, turn_df, full_text, measures)

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.error(f"Error in pos tag feature calculation: {e}")
    finally:
        return df_list

def get_sentiment(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        _, turn_list, full_text = text_list
        lemmatizer = spacy.load('en_core_web_sm')

        sentiment = SentimentIntensityAnalyzer()
        cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"], measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"]]

        for idx, u in enumerate(turn_list):
            try:
                
                sentiment_dict = sentiment.polarity_scores(u)
                mattrs = [get_mattr(u, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
                turn_df.loc[idx, cols] = list(sentiment_dict.values()) + mattrs
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                continue
                
        sentiment_dict = sentiment.polarity_scores(full_text)
        mattrs = [get_mattr(full_text, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]

        summ_df.loc[0, cols] = list(sentiment_dict.values()) + mattrs
        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.error(f"Error in sentiment feature calculation: {e}")
    finally:
        return df_list

def calculate_repetitions(words_texts, phrases_texts):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of repeated words and phrases in the input lists.

    Parameters:
    ...........
    words_texts: list
        List of transcribed text at the word level.
    phrases_texts: list
        List of transcribed text at the phrase level.

    Returns:
    ...........
    word_reps_perc: float
        The percentage of repeated words in the input lists.
    phrase_reps_perc: float
        The percentage of repeated phrases in the input lists.

    ------------------------------------------------------------------------------------------------------
    """
    # remove punctuation - lowercase
    words_texts = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in words_texts]
    phrases_texts = [phrase.translate(str.maketrans('', '', string.punctuation)).lower() for phrase in phrases_texts]

    # remove empty strings
    words_texts = [word for word in words_texts if word != '']
    phrases_texts = [phrase for phrase in phrases_texts if phrase != '']

    # sliding window of 10 words for repetitions
    if len(words_texts) <= 10:
        word_reps = len(words_texts) - len(set(words_texts))
        word_perc = 100*word_reps/len(words_texts)
    else:
        word_reps_list = []
        for i in range(len(words_texts) - 9):
            window = words_texts[i:i+10]
            word_reps_list.append(100*(len(window) - len(set(window))) / len(window))
        word_perc = np.mean(word_reps_list)

    if len(phrases_texts) == 0:
        return word_perc, np.nan
    
    # sliding window of 3 phrases for repetitions
    if len(phrases_texts) <= 3:
        phrase_reps = len(phrases_texts) - len(set(phrases_texts))
        phrase_perc = 100*phrase_reps/len(phrases_texts)
    else:
        phrase_reps_list = []
        for i in range(len(phrases_texts) - 3):
            window = phrases_texts[i:i+3]
            phrase_reps_list.append(100*(len(window) - len(set(window))) / len(window))
        phrase_perc = np.mean(phrase_reps_list)

    return word_perc, phrase_perc


def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, language, measures):
    """

    This function calculates the percentage of repeated words and phrases in the input text
    and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    utterances_speaker_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker
        after filtering out turns with less than min_turn_length words.
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    """
    
    try:
        word_df, turn_df, summ_df = df_list

        # turn-level
        if len(turn_df) > 0:
            for i in range(len(utterances_speaker_filtered)):
                row = utterances_speaker_filtered.iloc[i]
                words_texts = row[measures['words_texts']]
                phrases_texts = row[measures['phrases_texts']]

                word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

                turn_df.loc[i, measures['word_repeat_percentage']] = word_reps_perc
                turn_df.loc[i, measures['phrase_repeat_percentage']] = phrase_reps_perc

        # summary-level
        if len(turn_df) > 0:
            summ_df[measures['word_repeat_percentage']] = turn_df[measures['word_repeat_percentage']].mean(skipna=True)
            summ_df[measures['phrase_repeat_percentage']] = turn_df[measures['phrase_repeat_percentage']].mean(skipna=True)
        else:
            words_texts = [word for words in utterances_speaker[measures['words_texts']] for word in words]
            phrases_texts = [phrase for phrases in utterances_speaker[measures['phrases_texts']] for phrase in phrases]

            word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

            summ_df[measures['word_repeat_percentage']] = word_reps_perc
            summ_df[measures['phrase_repeat_percentage']] = phrase_reps_perc

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.error(f"Error in calculating repetitions: {e}")
    finally:
        return df_list

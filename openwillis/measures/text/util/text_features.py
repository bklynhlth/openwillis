# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# NLTK Tag list
TAG_DICT = {
    "PRP": "Pronoun",
    "PRP$": "Pronoun",
    "VB": "Verb",
    "VBD": "Verb",
    "VBG": "Verb",
    "VBN": "Verb",
    "VBP": "Verb",
    "VBZ": "Verb",
    "JJ": "Adjective",
    "JJR": "Adjective",
    "JJS": "Adjective",
    "NN": "Noun",
    "NNP": "Noun",
    "NNS": "Noun",
}


def get_tag(json_conf, tag_dict, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns an updated
     json_conf list with the part-of-speech tags.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    tag_dict: dict
        A dictionary mapping the NLTK tags to more readable tags.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    json_conf: list
        The updated json_conf list.

    ------------------------------------------------------------------------------------------------------
    """
    if len(json_conf) <= 0:
        return json_conf

    if "alternatives" not in json_conf[0].keys():
        # local vosk transcriber
        word_list = [word["word"] for word in json_conf if "word" in word]
    else:
        # aws transcriber
        word_list = [item["alternatives"][0]["content"] for item in json_conf]

    tag_list = nltk.pos_tag(word_list)

    for i, tag in enumerate(tag_list):
        if tag[1] in tag_dict.keys():
            json_conf[i][measures["tag"]] = tag_dict[tag[1]]
        else:
            json_conf[i][measures["tag"]] = "Other"

    return json_conf


def get_part_of_speech(df, tags, measures, index=0):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs,
     pronouns, adjectives, and nouns in the
     transcribed text, and adds them to the output dataframe df.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    tags: list
        A list of part-of-speech tags for the input text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    index: int
        The index of the row in the output dataframe df.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    if len(tags) == 0:
        return df

    df.loc[index, measures["speech_noun"]] = (
        100 * len(tags[tags == "Noun"]) / len(tags)
    )
    df.loc[index, measures["speech_verb"]] = (
        100 * len(tags[tags == "Verb"]) / len(tags)
    )
    df.loc[index, measures["speech_adj"]] = (
        100 * len(tags[tags == "Adjective"]) / len(tags)
    )
    df.loc[index, measures["speech_pronoun"]] = (
        100 * len(tags[tags == "Pronoun"]) / len(tags)
    )

    return df


def get_tag_summ(json_conf, df_list, text_indices, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs,
     pronouns, adjectives, and nouns in the
     transcribed text, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
            word_df, phrase_df, turn_df, summ_df
    text_indices: list
        List of indices for text_list.
            for phrases and turns.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """

    word_df, phrase_df, turn_df, summ_df = df_list
    phrase_index, turn_index = text_indices

    df_conf = pd.DataFrame(json_conf)

    # word-level analysis
    word_df[measures["part_of_speech"]] = df_conf[measures["tag"]]

    # phrase-level analysis
    for j, pindex in enumerate(phrase_index):
        prange = range(pindex[0], pindex[1] + 1)
        phrase_tags = df_conf.loc[df_conf[measures["old_index"]].isin(prange), measures["tag"]]

        phrase_df = get_part_of_speech(phrase_df, phrase_tags, measures, j)

    # turn-level analysis
    for j, uindex in enumerate(turn_index):
        urange = range(uindex[0], uindex[1] + 1)
        turn_tags = df_conf.loc[df_conf[measures["old_index"]].isin(urange), measures["tag"]]

        turn_df = get_part_of_speech(turn_df, turn_tags, measures, j)

    # file-level analysis
    summ_df = get_part_of_speech(summ_df, df_conf[measures["tag"]], measures)

    df_list = [word_df, phrase_df, turn_df, summ_df]

    return df_list


def get_mattr(word):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    word : list
        The input text as a list of words.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """
    filter_punc = list(value for value in word if value not in [".", "!", "?"])
    filter_punc = " ".join(str(filter_punc))
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=lex_richness.words)

    return mattr


def get_sentiment(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
            word_df, phrase_df, turn_df, summ_df
    text_list: list
        List of transcribed text.
            split into words, phrases, turns, and full text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, phrase_df, turn_df, summ_df = df_list
    word_list, phrase_list, turn_list, full_text = text_list

    sentiment = SentimentIntensityAnalyzer()

    # column names
    cols = [
        measures["neg"],
        measures["neu"],
        measures["pos"],
        measures["compound"],
        measures["speech_mattr"],
    ]

    # word-level analysis
    for idx, w in enumerate(word_list):
        try:
            sentiment_dict = sentiment.polarity_scores(w)

            word_df.loc[idx, cols[:-1]] = list(sentiment_dict.values())
        except Exception as e:
            logger.error(f"Error in sentiment analysis for word {w}: {e}")
            continue

    # phrase-level analysis
    for idx, p in enumerate(phrase_list):
        try:
            sentiment_dict = sentiment.polarity_scores(p)
            mattr = get_mattr(p)

            phrase_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]
        except Exception as e:
            logger.error(f"Error in sentiment analysis for phrase {p}: {e}")
            continue

    # turn-level analysis
    for idx, u in enumerate(turn_list):
        try:
            sentiment_dict = sentiment.polarity_scores(u)
            mattr = get_mattr(u)

            turn_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]
        except Exception as e:
            logger.error(f"Error in sentiment analysis for turn {u}: {e}")
            continue

    # file-level analysis
    sentiment_dict = sentiment.polarity_scores(full_text)
    mattr = get_mattr(full_text)

    summ_df.loc[0, cols] = list(sentiment_dict.values()) + [mattr]

    df_list = [word_df, phrase_df, turn_df, summ_df]

    return df_list


def get_sentiment_rater(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
            prompt_df, summ_df
    text_list: list
        List of transcribed text.
            split into turns, full text and prompt-adherent turns.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    prompt_df, summ_df = df_list
    turn_list, full_text, prompt_turn_list = text_list

    sentiment = SentimentIntensityAnalyzer()

    # column names
    cols = [
        measures["neg"],
        measures["neu"],
        measures["pos"],
        measures["compound"],
    ]

    # turn-level analysis
    for idx, u in enumerate(prompt_turn_list):
        try:
            sentiment_dict = sentiment.polarity_scores(u)

            prompt_df.loc[idx, cols] = list(sentiment_dict.values())
        except Exception as e:
            logger.error(f"Error in sentiment analysis for prompt {u}: {e}")
            continue

    # file-level analysis
    sentiment_dict = sentiment.polarity_scores(full_text)

    summ_df.loc[0, cols] = list(sentiment_dict.values())

    df_list = [prompt_df, summ_df]

    return df_list


def get_num_of_syllables(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the number of syllables in the input text.

    Parameters:
    ...........
    text: str
        The input text.

    Returns:
    ...........
    syllable_count: int
        The number of syllables in the input text.

    ---------------------------------------------------------------------------------------
    """

    syllable_tokenizer = nltk.tokenize.SyllableTokenizer()

    # remove punctuation
    punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    syllables = [syllable_tokenizer.tokenize(token) for token in nltk.word_tokenize(text) if token not in punctuation]
    # count the number of syllables in each word
    syllable_count = sum([len(token) for token in syllables])

    return syllable_count

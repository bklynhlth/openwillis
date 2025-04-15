# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages

import logging
import string
import unicodedata
from collections import Counter

import benepar
import nltk
import numpy as np
import pandas as pd
import spacy
from lexicalrichness import LexicalRichness
from nltk.tree import ParentedTree
from spacy.tokens import Span
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun",
            "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb", "DT": "Determiner"}
FIRST_PERSON_PRONOUNS = ["I", "me", "my", "mine", "myself"]
PRESENT = ["VBP", "VBZ"]
PAST = ["VBD", "VBN"]
MIN_WORDS = 10

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

def preprocess_text(text, lemmatizer):
    """
    Preprocess the input text: tokenize, remove punctuation, lowercase, lemmatize.

    Parameters:
    ...........
    text: str
            The input text to be analyzed.
    lemmatizer: spacy lemmatizer
            The lemmatizer to be used.

    Returns:
    ...........
    words_texts : list 
        List of lemmatized words.
    """
    words = nltk.word_tokenize(text)
    words = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in words]
    words = [w for w in words if w != '']
    words_texts = [token.lemma_ for token in lemmatizer(' '.join(words))]
    return words_texts

def get_brunet_index(text, lemmatizer, a=0.172):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates Brunet's Index to measure lexical diversity 
    of the input text.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.
    lemmatizer : spacy lemmatizer
        The lemmatizer to be used in the calculation.
    a : float
        A constant parameter for Brunet's Index (default is 0.172).

    Returns:
    ...........
    brunet : float
        The calculated Brunet's Index value.

    ------------------------------------------------------------------------------------------------------
    """
    words = preprocess_text(text, lemmatizer)
    
    N = len(words)       # Total number of words (tokens)
    
    if N < MIN_WORDS:
        return np.nan  # Avoid unreliable results
    
    V = len(set(words))  # Number of unique words (types)
    
    return N ** (V**-a)

def get_honore_statistic(text, lemmatizer):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates Honoré's Statistic to measure lexical richness 
    of the input text.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.
    lemmatizer : spacy lemmatizer
        The lemmatizer to be used in the calculation.

    Returns:
    ...........
    honore : float
        The calculated Honoré's Statistic value.

    ------------------------------------------------------------------------------------------------------
    """
    words = preprocess_text(text, lemmatizer)
    N = len(words)       # Total number of words (tokens)

    if N < MIN_WORDS:
        return np.nan  # Avoid unreliable results for small texts

    V = len(set(words))  # Number of unique words (types)

    freq_counts = Counter(words)
    V1 = sum(1 for word in freq_counts if freq_counts[word] == 1)  # Count of hapax legomena: words that occur once.

    if V == V1:  # Prevent division by zero
        return np.nan

    return 100 * (np.log(N) / (1 - (V1 / V)))

def get_parented_tree(span):
    """
    Converts a Benepar-generated parse string into an NLTK ParentedTree object.

    Parameters:
    -----------
    span : spacy.tokens.Span
        A spaCy span with a Benepar parse string in the ._.parse_string extension.

    Returns:
    --------
    nltk.tree.ParentedTree
        The corresponding ParentedTree object for syntactic traversal.
    """
    
    return ParentedTree.fromstring('(' + span._.parse_string + ')')

def yngve_score(tree):
    """
    Computes the Yngve score for a parse tree.
    
    The Yngve score quantifies syntactic complexity by increasing depth based on 
    the number of right siblings — capturing left-branching load.

    Parameters:
    -----------
    tree : nltk.tree.ParentedTree
        A syntactic parse tree.

    Returns:
    --------
    int
        Total Yngve score for the tree.
    """

    def compute(node, depth=0):
        if isinstance(node, str):
            return [depth]
        scores = []
        for i, child in enumerate(node):
            branch_depth = depth + (len(node) - i - 1)
            scores += compute(child, branch_depth)
        return scores
    return sum(compute(tree))

def frazier_score(tree):
    """
    Computes the Frazier score for a parse tree.

    The Frazier score quantifies memory load by increasing depth for right-branching nodes — 
    capturing processing complexity during sentence parsing.

    Parameters:
    -----------
    tree : nltk.tree.ParentedTree
        A syntactic parse tree.

    Returns:
    --------
    int
        Total Frazier score for the tree.
    """

    def compute(node, depth=0):
        if isinstance(node, str):
            return [depth]
        scores = []
        for i, child in enumerate(node):
            new_depth = depth if i == 0 else depth + 1
            scores += compute(child, new_depth)
        return scores
    return sum(compute(tree))

def clean_text(text):
    """
    Normalizes and replaces typographic characters for parsing compatibility.

    Parameters:
    -----------
    text : str
        The input text to clean.

    Returns:
    --------
    str
        Cleaned text with standard quotes, dashes, and apostrophes.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('’', "'").replace('‘', "'")
    return text

def score_excerpt(text):
    
    """
    Computes Yngve and Frazier syntactic complexity scores for a text excerpt.

    The function segments the excerpt into sentences, parses each sentence using Benepar, 
    and averages syntactic complexity scores across parse trees. Skips sentences that are 
    too long or unparseable.

    Parameters:
    -----------
    text : str
        The full excerpt to analyze.

    Returns:
    --------
    pandas.Series
        A series containing average Yngve and Frazier scores:
        {
            'yngve_score': float or NaN,
            'frazier_score': float or NaN
        }
    """

    nlp_splitter = spacy.load("en_core_web_sm")
    nlp_parser = spacy.load("en_core_web_sm")

    if "benepar" not in nlp_parser.pipe_names:
        nlp_parser.add_pipe("benepar", config={"model": "benepar_en3"})
    
    if not Span.has_extension("parse_tree"):
        Span.set_extension("parse_tree", getter=get_parented_tree)

    try:
        text = clean_text(text)

        if len(text.split()) < 8:
            # Too short to be meaningful
            return pd.Series({"yngve_score": np.nan, "frazier_score": np.nan})

        doc = nlp_splitter(text)  # Sentence segmentation only

        yngve_scores = []
        frazier_scores = []

        for sent in doc.sents:
            if len(sent) > 60:
                continue  # Skip very long sentences (too hard to parse reliably)

            try:
                # Parse the sentence with Benepar-enabled pipeline
                sent_doc = nlp_parser(sent.text)
                sents_parsed = list(sent_doc.sents)

                if not sents_parsed:
                    continue  # Sentence was not parsed properly

                tree = sents_parsed[0]._.parse_tree
                yngve_scores.append(yngve_score(tree))
                frazier_scores.append(frazier_score(tree))

            except Exception:
                # Skip this sentence on error (do not raise)
                continue

        if not yngve_scores:
            # No valid parses were obtained
            return pd.Series({"yngve_score": np.nan, "frazier_score": np.nan})

        return pd.Series({
            "yngve_score": np.mean(yngve_scores),
            "frazier_score": np.mean(frazier_scores)
        })

    except Exception:
        # If the entire excerpt fails (e.g. tokenizer error, encoding)
        return pd.Series({"yngve_score": np.nan, "frazier_score": np.nan})



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

    tag_list_verb = ["Present" if tag[1] in PRESENT else "Past" if tag[1] in PAST else "Other" for tag in tag_list]
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
        logger.info(f"Error in pos tag feature calculation: {e}")
    finally:
        return df_list

def get_sentiment(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframe summ_df.
    It also calculates measures of lexical diversity: the MATTR, Brunet's Index, Honoré's Statistic,
    and syntactic complexity: the Yngve and Frazier scores.

    The function has evolved so we could consider renaming it?

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

        cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"], 
                measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], 
                measures["speech_mattr_50"], measures["speech_mattr_100"], 
                measures["brunet_index"], measures["honore_statistic"], 
                measures["yngve_score"], measures["frazier_score"]]

        for idx, u in enumerate(turn_list):
            try:

                sentiment_dict = sentiment.polarity_scores(u)
                mattrs = [get_mattr(u, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
                brunet = get_brunet_index(u, lemmatizer)
                honore = get_honore_statistic(u, lemmatizer)
                syntax_scores = score_excerpt(u)
                yngve = syntax_scores["yngve_score"]
                frazier = syntax_scores["frazier_score"]

                turn_df.loc[idx, cols] = list(sentiment_dict.values()) + mattrs + [brunet, honore, yngve, frazier]

            except Exception as e:
                logger.info(f"Error in sentiment analysis: {e}")
                continue

        sentiment_dict = sentiment.polarity_scores(full_text)
        mattrs = [get_mattr(full_text, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
        brunet = get_brunet_index(full_text, lemmatizer)
        honore = get_honore_statistic(full_text, lemmatizer)
        syntax_scores = score_excerpt(full_text)
        yngve = syntax_scores["yngve_score"]
        frazier = syntax_scores["frazier_score"]

        summ_df.loc[0, cols] = list(sentiment_dict.values()) + mattrs + [brunet, honore, yngve, frazier]

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in sentiment feature calculation: {e}")
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
    def calculate_percentage_repetitions(text_list, window_size):
        """Helper function to calculate the percentage of repetitions in a sliding window."""
        if len(text_list) <= window_size:
            reps = len(text_list) - len(set(text_list))
            return 100 * reps / len(text_list) if len(text_list) > 0 else 0
        else:
            reps_list = [
                100 * (len(window) - len(set(window))) / len(window)
                for i in range(len(text_list) - window_size + 1)
                for window in [text_list[i:i + window_size]]
            ]
            return np.mean(reps_list)

    # Clean words and phrases: remove punctuation, convert to lowercase, and filter out empty strings
    words_texts = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in words_texts if word.strip()]
    phrases_texts = [phrase.translate(str.maketrans('', '', string.punctuation)).lower() for phrase in phrases_texts if phrase.strip()]

    # Calculate repetition percentages for words (sliding window of 10 words) and phrases (sliding window of 3 phrases)
    word_reps_perc = calculate_percentage_repetitions(words_texts, window_size=10)
    phrase_reps_perc = calculate_percentage_repetitions(phrases_texts, window_size=3) if phrases_texts else np.nan

    return word_reps_perc, phrase_reps_perc

def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures):
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
        if not turn_df.empty:
            for i in range(len(utterances_speaker_filtered)):
                row = utterances_speaker_filtered.iloc[i]
                words_texts = row[measures['words_texts']]
                phrases_texts = row[measures['phrases_texts']]

                word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

                turn_df.loc[i, measures['word_repeat_percentage']] = word_reps_perc
                turn_df.loc[i, measures['phrase_repeat_percentage']] = phrase_reps_perc

            # Calculate summary-level statistics
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
        logger.info(f"Error in calculating repetitions: {e}")
    finally:
        return df_list

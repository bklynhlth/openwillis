# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import re
import string
import logging

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_word_embeddings(word_list, tokenizer, model):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the word embeddings for the input text using BERT.

    Parameters:
    ...........
    word_list: list
        List of transcribed text at the word level.
    language: str
        Language of the transcribed text.

    Returns:
    ...........
    word_embeddings: numpy array
        The calculated word embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    if len(word_list) >= 512:
        # split the text into chunks of 512 tokens
        word_list = [word_list[i:i+512] for i in range(0, len(word_list), 512)]
        word_embeddings = []
        for chunk in word_list:
            inputs = tokenizer(chunk, return_tensors='pt', padding=True)
            outputs = model(**inputs)
            word_embeddings.append(outputs.last_hidden_state.mean(1).detach().numpy())
        word_embeddings = np.concatenate(word_embeddings, axis=0)
    else:
        inputs = tokenizer(word_list, return_tensors='pt', padding=True)
        outputs = model(**inputs)
        # Average pooling of the hidden states
        word_embeddings = outputs.last_hidden_state.mean(1).detach().numpy()
    return word_embeddings

def get_word_coherence_utterance(row, tokenizer, model, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures for a single utterance.

    Parameters:
    ...........
    row: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    tokenizer: BertTokenizer
        A tokenizer object for BERT.
    model: BertModel
        A BERT model object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_coherence: list
        A list containing the calculated semantic similarity of each word to the immediately preceding word.
    word_coherence_5: list
        A list containing the calculated semantic similarity of each word in 5-words window.
    word_coherence_10: list
        A list containing the calculated semantic similarity of each word in 10-words window.
    word_word_variability: dict
        A dictionary containing the calculated word-to-word variability at k inter-word distances.

    ------------------------------------------------------------------------------------------------------
    """
    words_texts = row[measures['words_texts']]
    if len(words_texts) == 0:
        # return empty lists if no words in the utterance
        return [np.nan]*len(words_texts), [np.nan]*len(words_texts), [np.nan]*len(words_texts), {k: [np.nan]*len(words_texts) for k in range(2, 11)}

    word_embeddings = get_word_embeddings(words_texts, tokenizer, model)
    similarity_matrix = cosine_similarity(word_embeddings)

    # calculate semantic similarity of each word to the immediately preceding word
    if len(words_texts) > 1:
        word_coherence = [np.nan] + [similarity_matrix[j, j-1] for j in range(1, len(words_texts))]
    else:
        word_coherence = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 5-words window
    if len(words_texts) > 5:
        word_coherence_5 = [np.nan]*2 + [np.mean(similarity_matrix[j-2:j+3, j]) for j in range(2, len(words_texts)-2)] + [np.nan]*2
    else:
        word_coherence_5 = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 10-words window
    if len(words_texts) > 10:
        word_coherence_10 = [np.nan]*5 + [np.mean(similarity_matrix[j-5:j+6, j]) for j in range(5, len(words_texts)-5)] + [np.nan]*5
    else:
        word_coherence_10 = [np.nan]*len(words_texts)

    # calculate word-to-word variability at k inter-word distances (for k from 2 to 10)
    # indicating semantic similarity between each word and the next following word at k inter-word distance
    word_word_variability = {}
    for k in range(2, 11):
        if len(words_texts) > k:
            word_word_variability[k] = [similarity_matrix[j, j+k] for j in range(len(words_texts)-k)] + [np.nan]*k
        else:
            word_word_variability[k] = [np.nan]*len(words_texts)

    return word_coherence, word_coherence_5, word_coherence_10, word_word_variability    

def get_word_coherence_summary(word_df, summ_df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function summarizes the word coherence measures at the summary level.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    summ_df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures['word_coherence_mean']] = word_df[measures['word_coherence']].mean(skipna=True)
    summ_df[measures['word_coherence_var']] = word_df[measures['word_coherence']].var(skipna=True)
    summ_df[measures['word_coherence_5_mean']] = word_df[measures['word_coherence_5']].mean(skipna=True)
    summ_df[measures['word_coherence_5_var']] = word_df[measures['word_coherence_5']].var(skipna=True)
    summ_df[measures['word_coherence_10_mean']] = word_df[measures['word_coherence_10']].mean(skipna=True)
    summ_df[measures['word_coherence_10_var']] = word_df[measures['word_coherence_10']].var(skipna=True)
    for k in range(2, 11):
        summ_df[measures[f'word_coherence_variability_{k}_mean']] = word_df[measures[f'word_coherence_variability_{k}']].mean(skipna=True)
        summ_df[measures[f'word_coherence_variability_{k}_var']] = word_df[measures[f'word_coherence_variability_{k}']].var(skipna=True)

    return summ_df

def append_nan_values(coherence_lists, row_len):
    """
    ------------------------------------------------------------------------------------------------------

    Helper function for appending NaN values to the coherence lists.

    Parameters:
    ...........
    coherence_lists: dict
        A dictionary containing the coherence lists.
    row_len: int
        The length of the row.

    Returns:
    ...........
    coherence_lists: dict
        The updated coherence lists.

    ------------------------------------------------------------------------------------------------------
    """
    coherence_lists['word_coherence'] += [np.nan] * row_len
    coherence_lists['word_coherence_5'] += [np.nan] * row_len
    coherence_lists['word_coherence_10'] += [np.nan] * row_len
    for k in range(2, 11):
        coherence_lists['variability'][k] += [np.nan] * row_len

    return coherence_lists

def get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    language: str
        Language of the transcribed text.
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

        # Initialize the appropriate model and tokenizer based on language
        if language in measures["english_langs"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')
        elif language in measures["supported_langs_bert"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = BertModel.from_pretrained('bert-base-multilingual-cased')
        else:
            logger.info(f"Language {language} not supported for word coherence analysis.")
            return df_list

        # Initialize coherence lists
        coherence_lists = {
            'word_coherence': [],
            'word_coherence_5': [],
            'word_coherence_10': [],
            'variability': {k: [] for k in range(2, 11)}
        }

        # Process each utterance
        for _, row in utterances_speaker.iterrows():
            try:
                if len(row[measures['words_texts']]) < min_coherence_turn_length:
                    coherence_lists = append_nan_values(coherence_lists, len(row[measures['words_texts']]))
                    continue

                # Get word coherence for the utterance
                coherence, coherence_5, coherence_10, variability = get_word_coherence_utterance(row, tokenizer, model, measures)

                # Append results to lists
                coherence_lists['word_coherence'] += coherence
                coherence_lists['word_coherence_5'] += coherence_5
                coherence_lists['word_coherence_10'] += coherence_10
                for k in range(2, 11):
                    coherence_lists['variability'][k] += variability[k]

            except Exception as e:
                logger.info(f"Error in word coherence analysis for row: {e}")
                coherence_lists = append_nan_values(coherence_lists, len(row[measures['words_texts']]))

        # Update word_df with calculated coherence values
        word_df[measures['word_coherence']] = coherence_lists['word_coherence']
        word_df[measures['word_coherence_5']] = coherence_lists['word_coherence_5']
        word_df[measures['word_coherence_10']] = coherence_lists['word_coherence_10']
        for k in range(2, 11):
            word_df[measures[f'word_coherence_variability_{k}']] = coherence_lists['variability'][k]

        # Update the summary-level dataframe
        summ_df = get_word_coherence_summary(word_df, summ_df, measures)

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in word coherence analysis: {e}")
    finally:
        return df_list

def calculate_log_probs(input_ids, masked_input_ids, model, start_offset, end_offset, idx):
    """
    ------------------------------------------------------------------------------------------------------

    Helper function to calculate log probabilities for a given window size

    Parameters:
    ...........
    input_ids: torch.Tensor
        The input tensor.
    masked_input_ids: torch.Tensor
        The masked input tensor.
    model: BertForMaskedLM
        A BERT model.
    start_offset: int
        The start offset.
    end_offset: int
        The end offset.
    idx: int
        The index.

    Returns:
    ...........
    float
        The calculated log probability.

    ------------------------------------------------------------------------------------------------------
    """
    input_ids_filtered = input_ids[:, start_offset:end_offset+1]
    masked_input_ids_filtered = masked_input_ids[:, start_offset:end_offset+1]
    idx_adjusted = idx - start_offset

    with torch.no_grad():
        outputs = model(input_ids=masked_input_ids_filtered, labels=input_ids_filtered)
    logits = outputs.logits[0, idx_adjusted].softmax(dim=0)
    true_log_prob = logits[input_ids[0, idx]].log().item()
    
    return true_log_prob

def calculate_perplexity(text, model, tokenizer):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pseudo-perplexity of the input text using BERT.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    model: BertForMaskedLM
        A BERT model.
    tokenizer: BertTokenizer
        A BERT tokenizer.

    Returns:
    ...........
    Tuple of floats
        The calculated pseudo-perplexity of the input text.
        The calculated pseudo-perplexity of the input text using 2 words before and after the masked token.
        The calculated pseudo-perplexity of the input text using 5 words before and after the masked token.
        The calculated pseudo-perplexity of the input text using 7 words before and after the masked token.

    ------------------------------------------------------------------------------------------------------
    """
    # Clean and tokenize the input text
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if len(clean_text) == 0 or len(clean_text.split()) < 2:
        return np.nan

    tokens = tokenizer(clean_text, return_tensors='pt')
    input_ids = tokens.input_ids
    masked_input_ids = input_ids.clone()

    log_probs_dict = {2: [], 5: [], 7: [], 256: []}

    # Iterate over each token in the input text
    for i in range(input_ids.size(1)):
        masked_input_ids[0, i] = tokenizer.mask_token_id

        # Calculate log probabilities for different window sizes
        for window_size in [256, 2, 5, 7]:
            start = max(0, i - window_size)
            end = min(input_ids.size(1), i + window_size)
            log_probs_dict[window_size].append(calculate_log_probs(input_ids, masked_input_ids, model, start, end, i))

        # Unmask the token for the next iteration
        masked_input_ids[0, i] = input_ids[0, i]

    # Calculate perplexity for each window size
    perplexity = np.exp(-np.mean(log_probs_dict[256]))
    perplexity_5 = np.exp(-np.mean(log_probs_dict[2]))
    perplexity_11 = np.exp(-np.mean(log_probs_dict[5]))
    perplexity_15 = np.exp(-np.mean(log_probs_dict[7]))

    return perplexity, perplexity_5, perplexity_11, perplexity_15

def calculate_phrase_tangeniality(phrases_texts, utterance_text, sentence_encoder, bert, tokenizer):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the semantic similarity of each phrase to the immediately preceding phrase,
    the semantic similarity of each phrase to the phrase 2 turns before, and the pseudo-perplexity of the turn.

    Parameters:
    ...........
    phrases_texts: list
        List of transcribed text at the phrase level.
    utterance_text: str
        The full transcribed text.
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    bert: BertForMaskedLM
        A BERT model.
    tokenizer: BertTokenizer
        A BERT tokenizer.

    Returns:
    ...........
    sentence_tangeniality1: float
        The semantic similarity of each phrase to the immediately preceding phrase.
    sentence_tangeniality2: float
        The semantic similarity of each phrase to the phrase 2 turns before.
    perplexity: float
        The pseudo-perplexity of the turn.
    perplexity_5: float
        The pseudo-perplexity of the turn using 2 words before and after the masked token.
    perplexity_11: float
        The pseudo-perplexity of the turn using 5 words before and after the masked token.
    perplexity_15: float
        The pseudo-perplexity of the turn using 7 words before and after the masked token.

    ------------------------------------------------------------------------------------------------------
    """
    sentence_tangeniality1 = np.nan
    sentence_tangeniality2 = np.nan
    if sentence_encoder is not None and len(phrases_texts) > 0:
        phrase_embeddings = sentence_encoder.encode(phrases_texts)
        similarity_matrix = cosine_similarity(phrase_embeddings)

        # calculate semantic similarity of each phrase to the immediately preceding phrase
        if len(phrases_texts) > 1:
            sentence_tangeniality1 = np.mean([similarity_matrix[j-1, j] for j in range(1, len(phrases_texts))])

        # calculate semantic similarity of each phrase to the phrase 2 turns before
        if len(phrases_texts) > 2:
            sentence_tangeniality2 = np.mean([similarity_matrix[j-2, j] for j in range(2, len(phrases_texts))])

    perplexity, perplexity_5, perplexity_11, perplexity_15 = np.nan, np.nan, np.nan, np.nan
    if tokenizer is not None and bert is not None:
        # calculate pseudo-perplexity of the turn and indicating how predictable the turn is
        perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_perplexity(utterance_text, bert, tokenizer)        

    return sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15

def calculate_slope(y):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the slope
     of the input list using linear regression

    Parameters:
    ...........
    y: list
        A list of values.

    Returns:
    ...........
    float
        The calculated slope of the input list.

    ------------------------------------------------------------------------------------------------------
    """
    # remove NaNs
    y = [val for val in y if not np.isnan(val)]

    x = range(len(y))
    try:
        slope, _ = np.polyfit(x, y, 1)
    except:
        slope = np.nan

    return slope

def init_model(language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function initializes the appropriate models and tokenizers based on language.

    Parameters:
    ...........
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    tokenizer: BertTokenizer
        A BERT tokenizer.
    bert: BertForMaskedLM
        A BERT model.

    ------------------------------------------------------------------------------------------------------
    """
    sentence_encoder, tokenizer, bert = None, None, None
    if language in measures["english_langs"]:
        sentence_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bert = BertForMaskedLM.from_pretrained('bert-base-cased')
    else:
        if language in measures["supported_langs_sentence_embeddings"]:
            sentence_encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        else:
            logger.info(f"Language {language} not supported for phrase coherence analysis")

        if language in measures["supported_langs_bert"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            bert = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        else:
            logger.info(f"Language {language} not supported for perplexity analysis")

    return sentence_encoder, tokenizer, bert

def calculate_turn_coherence(utterances_filtered, turn_df, min_coherence_turn_length, speaker_label, sentence_encoder, bert, tokenizer, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures for the specified speaker.

    Parameters:
    ...........
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    bert: BertForMaskedLM
        A BERT model.
    tokenizer: BertTokenizer
        A BERT tokenizer.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # semantic similarity between each pair of utterances
    utterances_texts = utterances_filtered[measures['utterance_text']].values.tolist()
    utterances_embeddings = sentence_encoder.encode(utterances_texts) if sentence_encoder else None
    similarity_matrix = cosine_similarity(utterances_embeddings) if sentence_encoder else None

    # Initialize coherence lists
    sentence_tangeniality1_list, sentence_tangeniality2_list = [], []
    perplexity_list, perplexity_5_list, perplexity_11_list, perplexity_15_list = [], [], [], []
    turn_to_turn_tangeniality_list = []
    for i, row in utterances_filtered.iterrows():
        current_speaker = row[measures['speaker_label']]

        if current_speaker != speaker_label:
            continue
        elif len(row[measures['words_texts']]) < min_coherence_turn_length:
            sentence_tangeniality1_list.append(np.nan)
            sentence_tangeniality2_list.append(np.nan)
            perplexity_list.append(np.nan)
            perplexity_5_list.append(np.nan)
            perplexity_11_list.append(np.nan)
            perplexity_15_list.append(np.nan)
            turn_to_turn_tangeniality_list.append(np.nan)
            continue

        phrases_texts = row[measures['phrases_texts']]
        utterance_text = row[measures['utterance_text']]
        
        sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_phrase_tangeniality(
            phrases_texts, utterance_text, sentence_encoder, bert, tokenizer
        )
        
        sentence_tangeniality1_list.append(sentence_tangeniality1)
        sentence_tangeniality2_list.append(sentence_tangeniality2)
        perplexity_list.append(perplexity)
        perplexity_5_list.append(perplexity_5)
        perplexity_11_list.append(perplexity_11)
        perplexity_15_list.append(perplexity_15)

        if i == 0 or len(utterances_filtered.iloc[i - 1][measures['words_texts']]) < min_coherence_turn_length or not sentence_encoder:
            turn_to_turn_tangeniality_list.append(np.nan)
        else:
            turn_to_turn_tangeniality_list.append(similarity_matrix[i, i - 1])

    turn_df[measures['sentence_tangeniality1']] = sentence_tangeniality1_list
    turn_df[measures['sentence_tangeniality2']] = sentence_tangeniality2_list
    turn_df[measures['perplexity']] = perplexity_list
    turn_df[measures['perplexity_5']] = perplexity_5_list
    turn_df[measures['perplexity_11']] = perplexity_11_list
    turn_df[measures['perplexity_15']] = perplexity_15_list
    turn_df[measures['turn_to_turn_tangeniality']] = turn_to_turn_tangeniality_list

    return turn_df

def get_phrase_coherence(df_list, utterances_filtered, min_coherence_turn_length, speaker_label, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    language: str
        Language of the transcribed text.
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

        sentence_encoder, tokenizer, bert = init_model(language, measures)
        if not sentence_encoder and not tokenizer and not bert:
            logger.info(f"Language {language} not supported for phrase coherence nor perplexity analysis")
            return df_list

        # turn-level
        if len(turn_df) > 0:
            turn_df = calculate_turn_coherence(utterances_filtered, turn_df, min_coherence_turn_length, speaker_label, sentence_encoder, bert, tokenizer, measures)

            for measure in ['sentence_tangeniality1', 'sentence_tangeniality2', 'perplexity', 'perplexity_5', 'perplexity_11', 'perplexity_15', 'turn_to_turn_tangeniality']:
                if turn_df[measures[measure]].isnull().all():
                    continue
                summ_df[measures[measure + '_mean']] = turn_df[measures[measure]].mean(skipna=True)
                summ_df[measures[measure + '_var']] = turn_df[measures[measure]].var(skipna=True)

            if not turn_df[measures['turn_to_turn_tangeniality']].isnull().all():
                summ_df[measures['turn_to_turn_tangeniality_slope']] = calculate_slope(turn_df[measures['turn_to_turn_tangeniality']])

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in phrase coherence analysis: {e}")
    finally:
        return df_list

# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages

import numpy as np
import logging
from scipy import spatial

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_similarity_matrix(embeddings1, embeddings2):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the cosine similarity between
        two sets of embeddings.

    Parameters:
    ...........
    embeddings1: numpy array
        A numpy array containing the embeddings of the first set of text.
    embeddings2: numpy array
        A numpy array containing the embeddings of the second set of text.

    Returns:
    ...........
    similarity_matrix: numpy array
        A numpy array containing the cosine similarity between the two sets of embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))

    for i in range(len(embeddings1)):
        for j in range(len(embeddings2)):
            similarity_matrix[i, j] = 1 - spatial.distance.cosine(embeddings1[i], embeddings2[j])

    return similarity_matrix


def text_adherence_prompt(prompt_df, similarity_matrix, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the prompt adherence of the input text.
        and adds them to the output dataframe prompt_df.

    Parameters:
    ...........
    prompt_df: pandas dataframe
        A dataframe containing prompt summary information
    similarity_matrix: numpy array
        A numpy array containing the cosine similarity between the two sets of embeddings.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    prompt_df: pandas dataframe
        The updated prompt_df dataframe.
    turn_ids: numpy array
        A numpy array containing the indices of the turns that are prompt-adherent.

    ------------------------------------------------------------------------------------------------------
    """

    # calculate the maximum similarity score for each prompt
    prompt_similarities = np.max(similarity_matrix, axis=0)
    prompt_matchings = np.argmax(similarity_matrix, axis=0)

    prompt_df[measures["prompt_adherence"]] = prompt_similarities

    return prompt_df, prompt_matchings


def text_adherence_summ(summ_df, prompt_df, similarity_matrix, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function updates the summary dataframe summ_df with prompt adherence information.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing phrase summary information
    prompt_df: pandas dataframe
        A dataframe containing prompt summary information
    similarity_matrix: numpy array
        A numpy array containing the cosine similarity between the two sets of embeddings.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # number of prompts
    ## could add a check here for the number of prompts being
    ## rows with prompt_adherence > some threshold
    summ_df[measures["no_prompts"]] = [len(prompt_df)]
    # average prompt adherence
    summ_df[measures["mean_prompt_adherence"]] = prompt_df[measures["prompt_adherence"]].mean()
    # percentage of turns being prompt-adherent
    summ_df[measures["promps_turns_percentage"]] = 100 * len(prompt_df) / len(similarity_matrix)
    
    return summ_df


def get_text_adherence(df_list, text_list, text_indices, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the prompt adherence of the input text.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
            prompt_df, summ_df
    text_list: list
        List of transcribed text.
            split into turns, full text, and prompt list.
    text_indices: list
        List of indices for text_list.
            for turns and prompts (ids).
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.
            prompt_df, summ_df
    text_list: list
        List of updated transcribed text.
            split into turns, full text, and prompt-adherent turns.
    text_indices: list
        List of updated indices for text_list.
            for turns, prompts (ids), and prompt-adherent turns.

    ------------------------------------------------------------------------------------------------------
    """
    prompt_df, summ_df = df_list
    turn_list, full_text, prompt_list = text_list
    turn_indices = text_indices[0]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    turn_embeddings = model.encode(turn_list)
    prompt_embeddings = model.encode(prompt_list)

    similarity_matrix = get_similarity_matrix(turn_embeddings, prompt_embeddings)

    prompt_df, turn_ids = text_adherence_prompt(prompt_df, similarity_matrix, measures)

    summ_df = text_adherence_summ(summ_df, prompt_df, similarity_matrix, measures)

    # update prompt_list with matched prompts
    turn_prompt_list = [turn_list[i] for i in turn_ids]
    prompt_indices = [turn_indices[i] for i in turn_ids]

    df_list = [prompt_df, summ_df]
    text_list[2] = turn_prompt_list
    text_indices.append(prompt_indices)

    return df_list, text_list, text_indices

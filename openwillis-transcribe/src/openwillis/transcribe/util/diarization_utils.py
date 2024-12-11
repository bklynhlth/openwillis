# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
from enum import Enum
from multiprocessing import Pool
import time
import random
import logging
from functools import wraps

import boto3
import numpy as np
from scipy import optimize

# avoid warning about tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_FORMAT = """### Instruction
In the speaker diarization transcript below, some words are potentially misplaced. Please correct those words and move them to the right speaker. Directly show the corrected transcript without explaining what changes were made or why you made those changes.:

{{ user_msg_1 }}

### Answer

"""


def exponential_backoff_decorator(max_retries, base_delay):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result_func = func(*args, **kwargs)
                    return result_func
                except Exception as e:
                    retries += 1
                    delay = base_delay * 2**retries + random.uniform(0, 1)
                    logging.error(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
            raise Exception("Max retries reached, operation failed.")

        return wrapper

    return decorator

### Functions taken from https://github.com/google/speaker-id/tree/master/DiarizationLM
### Slightly modified to fit OpenWillis
def create_diarized_text(word_labels, speaker_labels):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a diarized text from word and speaker labels.

    Parameters:
    ...........
    word_labels: list
        List of words.
    speaker_labels: list
        List of speaker labels.

    Returns:
    ...........
    str: Diarized text.
     e.g. "<spk:1> hello <spk:2> how are you <spk:1> I am fine"

    ------------------------------------------------------------------------------------------------------
    """
    output = []
    previous_speaker = None

    for word, speaker in zip(word_labels, speaker_labels):
        if speaker != previous_speaker:
            output.append("<spk:" + speaker + ">")

        output.append(word)
        previous_speaker = speaker

    return " ".join(output)


def extract_text_and_spk(diarized_text):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts text and speaker labels from diarized text.

    Parameters:
    ...........
    diarized_text: str
        Previously diarized string.

    Returns:
    ...........
    str: words concatenated with spaces.
    str: speaker labels concatenated with spaces.

    ------------------------------------------------------------------------------------------------------
    """
    spk = "1"
    previous_spk = "1"
    result_text = []
    result_spk = []

    for word in diarized_text.split():
        if word.startswith("<spk:"):
            if not word.endswith(">"):
                word += ">"
            spk = word[len("<spk:") : -len(">")]
            # Handle undefined behaviors of non-recognizable spk with a placeholder.
            try:
                spk_int = int(spk)
                if not spk or spk_int < 1 or spk_int > 10:
                    raise ValueError("Seeing unexpected word: ", word)
                previous_spk = spk
            except ValueError:
                print("Skipping meaningless speaker token:", word)
                spk = previous_spk
        else:
            result_text.append(word)
            result_spk.append(spk)

    return " ".join(result_text), " ".join(result_spk)


class EditOp(Enum):
    Correct = 0
    Substitution = 1
    Insertion = 2
    Deletion = 3


# Computes the Levenshtein alignment between strings ref and hyp, where the
# tokens in each string are separated by delimiter.
# Outputs a tuple : (edit_distance, alignment) where
# alignment is a list of pairs (ref_pos, hyp_pos) where ref_pos is a position
# in ref and hyp_pos is a position in hyp.
# As an example, for strings 'a b' and 'a c', the output would look like:
# (1, [(0,0), (1,1)]
# Note that insertions are represented as (-1, j) and deletions as (i, -1).
def levenshtein_with_edits(ref, hyp, delimiter=" ", print_debug_info=False):
    align = []
    s1 = ref.split(delimiter)
    s2 = hyp.split(delimiter)
    n1 = len(s1)
    n2 = len(s2)
    costs = np.zeros((n1 + 1, n2 + 1), dtype=np.int16)
    backptr = np.zeros((n1 + 1, n2 + 1), dtype=EditOp)

    for i in range(n1 + 1):  # ref
        costs[i][0] = i  # deletions

    for j in range(n2):  # hyp
        costs[0][j + 1] = j + 1  # insertions
        for i in range(n1):  # ref
            # (i,j) <- (i,j-1)
            ins = costs[i + 1][j] + 1
            # (i,j) <- (i-1,j)
            del_ = costs[i][j + 1] + 1
            # (i,j) <- (i-1,j-1)
            sub = costs[i][j] + (s1[i] != s2[j])
            costs[i + 1][j + 1] = min(ins, del_, sub)
            if costs[i + 1][j + 1] == ins:
                backptr[i + 1][j + 1] = EditOp.Insertion
            elif costs[i + 1][j + 1] == del_:
                backptr[i + 1][j + 1] = EditOp.Deletion
            elif s1[i] == s2[j]:
                backptr[i + 1][j + 1] = EditOp.Correct
            else:
                backptr[i + 1][j + 1] = EditOp.Substitution

    if print_debug_info:
        print("Mincost: ", costs[n1][n2])
    i = n1
    j = n2
    # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
    # is a position in n2.
    while i > 0 or j > 0:
        if print_debug_info:
            print("i: ", i, " j: ", j)
        ed_op = EditOp.Correct
        if i >= 0 and j >= 0:
            ed_op = backptr[i][j]
        if i >= 0 and j < 0:
            ed_op = EditOp.Deletion
        if i < 0 and j >= 0:
            ed_op = EditOp.Insertion
        if i < 0 and j < 0:
            raise RuntimeError("Invalid alignment")
        if ed_op == EditOp.Insertion:
            align.append((-1, j - 1))
            j -= 1
        elif ed_op == EditOp.Deletion:
            align.append((i - 1, -1))
            i -= 1
        else:
            align.append((i - 1, j - 1))
            i -= 1
            j -= 1

    align.reverse()
    return costs[n1][n2], align


def normalize_text(text):
    """Normalize text."""
    # Convert to lower case.
    text_lower = text.lower()
    # Remove punctuation.
    text_de_punt = text_lower.replace(",", "").replace(".", "").replace("_", "").strip()
    if len(text_lower.split()) == len(text_de_punt.split()):
        return text_de_punt
    else:
        # If ater removing punctuation, we dropped words, then we keep punctuation.
        return text_lower


def get_aligned_hyp_speakers(
    hyp_text, ref_text, ref_spk,
    print_debug_info=False,
):
    """Align ref_text to hyp_text, then apply the alignment to ref_spk."""
    # Counters for insertions and deletions in hyp and ref text alignment.
    num_insertions, num_deletions = 0, 0

    # Get the alignment.
    _, align = levenshtein_with_edits(
        normalize_text(ref_text), normalize_text(hyp_text)
    )

    ref_spk_list = ref_spk.split()
    hyp_spk_align = []

    # Apply the alignment on ref speakers.
    for i, j in align:
        if i == -1:
            # hyp has insertion
            hyp_spk_align.append("-1")
            num_insertions += 1
        elif j == -1:
            # hyp has deletion
            num_deletions += 1
            continue
        else:
            hyp_spk_align.append(ref_spk_list[i])
    hyp_spk_align = " ".join(hyp_spk_align)

    if print_debug_info:
        print("Number of insertions: ", num_insertions)
        print("Number of deletions: ", num_deletions)
        # This is not the traditional denominator of WER. Instead, this is
        # len(hyp) + len(ref) - len(SUB).
        print("Length of align pairs: ", len(align))
    return hyp_spk_align


def get_oracle_speakers(hyp_spk, hyp_spk_align):
    """Get the oracle speakers for hypothesis."""
    hyp_spk_list = [int(x) for x in hyp_spk.split()]
    hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

    # Build cost matrix.
    max_spk = max(max(hyp_spk_list), max(hyp_spk_align_list))
    cost_matrix = np.zeros((max_spk, max_spk))
    for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
        cost_matrix[aligned - 1, original - 1] += 1

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(cost_matrix, maximize=True)

    # Build oracle.
    hyp_spk_oracle = hyp_spk_list.copy()
    for i in range(len(hyp_spk_list)):
        if hyp_spk_align_list[i] == -1:
            # There are some missing words. In such cases, we just use the original
            # speaker for these words if possible.
            if hyp_spk_list[i] == -1:
                # If we don't have original speaker for missing words, just use the
                # previous speaker if possible.
                # This is useful for the update_hyp_text_in_utt_dict() function.
                if i == 0:
                    hyp_spk_oracle[i] = 1
                else:
                    hyp_spk_oracle[i] = hyp_spk_oracle[i - 1]
            continue
        assert row_index[hyp_spk_align_list[i] - 1] == hyp_spk_align_list[i] - 1
        hyp_spk_oracle[i] = col_index[hyp_spk_align_list[i] - 1] + 1

    return hyp_spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def transcript_preserving_speaker_transfer(src_text, src_spk, tgt_text, tgt_spk):
    """Apply source speakers to target."""

    if len(tgt_text.split()) != len(tgt_spk.split()):
        raise ValueError("tgt_text and tgt_spk must have the same length")

    if len(src_text.split()) != len(src_spk.split()):
        raise ValueError("src_text and src_spk must have the same length")

    tgt_spk_align = get_aligned_hyp_speakers(
        hyp_text=tgt_text,
        ref_text=src_text,
        ref_spk=src_spk,
    )
    oracle_speakers = get_oracle_speakers(hyp_spk=tgt_spk, hyp_spk_align=tgt_spk_align)

    return " ".join([str(x) for x in oracle_speakers])
###


def extract_transcription_aws(transcript_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the speaker labels and words from an AWS transcription JSON.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.

    Returns:
    ...........
    labels_aws: list
        List of speaker labels.
    words_aws: list
        List of transcribed words.
    translate_json: dict
        Dictionary to translate '1' and '2' to the original speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    labels_aws = []
    words_aws = []

    for item in transcript_json["results"]["items"]:
        if item["type"] == "pronunciation":
            labels_aws.append(item["speaker_label"])
            words_aws.append(item["alternatives"][0]["content"].lower())

    # get unique speaker labels
    labels_aws_set = list(set(labels_aws))
    if len(list(set(labels_aws))) != 2:
        raise ValueError("Only two speakers supported")

    translate_json = {labels_aws_set[0]: "1", labels_aws_set[1]: "2"}
    # transform speaker labels to '1' and '2'
    for i, label in enumerate(labels_aws):
        labels_aws[i] = translate_json[label]

    # opposite to recreate original speaker labels
    translate_json = {"1": labels_aws_set[0], "2": labels_aws_set[1]}
    return labels_aws, words_aws, translate_json


def extract_transcription_whisperx(transcript_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the speaker labels and words from a WhisperX transcribed JSON.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.

    Returns:
    ...........
    labels_whisperx: list
        List of speaker labels.
    words_whisperx: list
        List of transcribed words.
    translate_json: dict
        Dictionary to translate '1' and '2' to the original speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    labels_whisperx = []
    words_whisperx = []

    for item in transcript_json["segments"]:
        labels_whisperx += [item["speaker"] for _ in range(len(item["words"]))]
        words_whisperx += [x["word"].lower() for x in item["words"]]

    # get unique speaker labels
    labels_whisperx_set = list(set(labels_whisperx))
    if len(list(set(labels_whisperx))) != 2:
        raise ValueError("Only two speakers supported")

    translate_json = {labels_whisperx_set[0]: "1", labels_whisperx_set[1]: "2"}
    # transform speaker labels to '1' and '2'
    for i, label in enumerate(labels_whisperx):
        labels_whisperx[i] = translate_json[label]

    # opposite of translate_json
    translate_json = {"1": labels_whisperx_set[0], "2": labels_whisperx_set[1]}
    return labels_whisperx, words_whisperx, translate_json


def modify_transcription_aws(transcript_json, corrected_labels):
    """
    ------------------------------------------------------------------------------------------------------

    This function modifies the speaker labels in an AWS transcription JSON
        using the corrected labels.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    corrected_labels: list
        List of corrected speaker labels.

    Returns:
    ...........
    dict: JSON response object with corrected speaker labels.

    ------------------------------------------------------------------------------------------------------
    """

    for item in transcript_json["results"]["items"]:
        if item["type"] == "pronunciation":
            new_label = corrected_labels.pop(0)
            item["speaker_label"] = new_label
        else:
            item["speaker_label"] = new_label

    return transcript_json


def modify_transcription_whisperx(transcript_json, corrected_labels):
    """
    ------------------------------------------------------------------------------------------------------

    This function modifies the speaker labels in an WhisperX transcription JSON
        using the corrected labels.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    corrected_labels: list
        List of corrected speaker labels.

    Returns:
    ...........
    dict: JSON response object with corrected speaker labels.

    ------------------------------------------------------------------------------------------------------
    """

    res = []
    words = []
    starts = []
    ends = []

    for item in transcript_json["segments"]:
        for x in item["words"]:
            words.append(x["word"])
            starts.append(x["start"])
            ends.append(x["end"])

    temp = {"words": []}

    for i in range(len(words)):
        if i > 0 and corrected_labels[i] != corrected_labels[i - 1]:

            temp["start"] = temp["words"][0]["start"]
            temp["end"] = temp["words"][-1]["end"]
            temp["speaker"] = temp["words"][0]["speaker"]
            temp["text"] = " ".join([x["word"] for x in temp["words"]])

            res.append(temp)

            temp = {"words": []}

            if i == len(words) - 1:
                temp["words"].append(
                    {
                        "word": words[i],
                        "start": starts[i],
                        "end": ends[i],
                        "speaker": corrected_labels[i],
                    }
                )
                temp["start"] = temp["words"][0]["start"]
                temp["end"] = temp["words"][-1]["end"]
                temp["speaker"] = temp["words"][0]["speaker"]
                temp["text"] = " ".join([x["word"] for x in temp["words"]])

                res.append(temp)

                temp = {"words": []}

        elif i == len(words) - 1 and corrected_labels[i] == corrected_labels[i - 1]:

            temp["words"].append(
                {
                    "word": words[i],
                    "start": starts[i],
                    "end": ends[i],
                    "speaker": corrected_labels[i],
                }
            )
            temp["start"] = temp["words"][0]["start"]
            temp["end"] = temp["words"][-1]["end"]
            temp["speaker"] = temp["words"][0]["speaker"]
            temp["text"] = " ".join([x["word"] for x in temp["words"]])

            res.append(temp)

        temp["words"].append(
            {
                "word": words[i],
                "start": starts[i],
                "end": ends[i],
                "speaker": corrected_labels[i],
            }
        )

    return {"segments": res}


def split_transcription(words, speakers, character_limit=2250):
    """
    ------------------------------------------------------------------------------------------------------

    This function splits a transcription into chunks of predefined character limit.
    
    Parameters:
    ...........
    words:
        List of transcribed words.
    speakers:
        List of speaker labels.
    character_limit:
        Maximum number of characters in each chunk.

    Returns:
    ...........
    prompts:
        List of diarized text chunks.
    """

    prompts = []
    current_chunk_words = []
    current_chunk_speakers = []
    current_chunk_length = 0

    # split the transcription into chunks based on character limit
    for word, speaker in zip(words, speakers):
        if current_chunk_length + len(word) + 1 > character_limit:  # +1 for space or punctuation
            prompts.append(create_diarized_text(current_chunk_words, current_chunk_speakers))
            current_chunk_words = []
            current_chunk_speakers = []
            current_chunk_length = 0

        current_chunk_words.append(word)
        current_chunk_speakers.append(speaker)
        current_chunk_length += len(word) + 1  # +1 for space or punctuation

    if current_chunk_words:  # Add the last chunk if it exists
        prompts.append(create_diarized_text(current_chunk_words, current_chunk_speakers))

    return prompts


def apply_formatting(chunk):
    """
    ------------------------------------------------------------------------------------------------------

    This function applies the formatting to a diarized text chunk.

    Parameters:
    ...........
    chunk: str
        Diarized text chunk.

    Returns:
    ...........
    dict: Input data for the API call.

    ------------------------------------------------------------------------------------------------------
    """
    prompt = PROMPT_FORMAT.replace("{{ user_msg_1 }}", chunk)
    res = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":2048, "top_p":0.5, "temperature":0.2, "stop":["</s>", "###"]
        }
    }
    return res


def preprocess_str(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function preprocesses a string by removing punctuation and multiple spaces.

    Parameters:
    ...........
    text: str
        Input string.

    Returns:
    ...........
    str: Preprocessed string.

    ------------------------------------------------------------------------------------------------------
    """
    text = text.replace("\n", " ").replace("\\", "")
    # remove punctuation - except for "'"
    text = "".join([c for c in text if c.isalnum() or c in [" ", "'", "<", ">", ":"]])
    # remove multiple spaces
    text = " ".join(text.split())
    text = text.lower().strip()
    return text


def extract_predicted_speakers(prompts, results, translate_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the predicted speakers from the outputs of the speaker diarization model.

    Parameters:
    ...........
    prompts: dict
        Dictionary of diarized text chunks.
        key: chunk index, value: diarized text chunk.
    results: dict
        Dictionary of model outputs.
        key: chunk index, value: model output.
    translate_json: dict
        Dictionary to translate '1' and '2' to the original speaker labels.

    Returns:
    ...........
    speakers_pred: list
        List of predicted speaker labels for the entire transcription.

    ------------------------------------------------------------------------------------------------------
    """
    speakers_pred = []

    for idx in sorted(results.keys()):
        input = prompts[idx]
        output = results[idx].split("### Answer\n")[1].strip()

        input = preprocess_str(input)
        output = preprocess_str(output)

        # extract text and speaker
        words_in, speakers_in = extract_text_and_spk(input)
        words_out, speakers_out = extract_text_and_spk(output)

        if len(output) > 1.5 * len(input):
            # if the output is significantly longer than the input,
            # it is likely the model has hallucinated some words
            # thus, we will not apply the correction
            speakers_pred += [translate_json[x] for x in speakers_in.split()]
            continue

        speakers_out2 = transcript_preserving_speaker_transfer(
            words_out, speakers_out, words_in, speakers_in
        )

        speakers_pred += [translate_json[x] for x in speakers_out2.split()]

    return speakers_pred


def extract_prompts(transcript_json, asr):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the prompts and the translation JSON from a transcription JSON.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    asr: str
        Automatic Speech Recognition (ASR)
         system used to transcribe the audio.

    Returns:
    ...........
    prompts: dict
        Dictionary of diarized text chunks.
        key: chunk index, value: diarized text chunk.
    translate_json: dict
        Dictionary to translate '1' and '2' to the original speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    if asr == "aws":
        speakers, words, translate_json = extract_transcription_aws(transcript_json)
    elif asr == "whisperx":
        speakers, words, translate_json = extract_transcription_whisperx(
            transcript_json
        )
    else:
        raise ValueError("ASR not supported")

    chunks = split_transcription(words, speakers)

    prompts = {idx: chunk for idx, chunk in enumerate(chunks)}

    return prompts, translate_json


class AWSProcessor:
    def __init__(self, endpoint_name, input_param):
        self.endpoint_name = endpoint_name
        self.input_param = input_param

    @exponential_backoff_decorator(max_retries=3, base_delay=90)
    def process_chunk(self, args):
        """
        ------------------------------------------------------------------------------------------------------
        This function processes a diarized text chunk using the SageMaker endpoint;
        it is used for parallel processing.

        Parameters:
        ...........
        args: tuple
            Tuple of arguments.
            idx: int, chunk index.
            prompt: str, diarized text chunk.

        Returns:
        ...........
        tuple: Tuple of chunk index and model output.

        ------------------------------------------------------------------------------------------------------
        """
        idx, prompt = args

        # initialize boto3 client
        if self.input_param['access_key'] and self.input_param['secret_key']:
            client = boto3.client('sagemaker-runtime', region_name=self.input_param['region'], 
                                        aws_access_key_id=self.input_param['access_key'], 
                                        aws_secret_access_key=self.input_param['secret_key'])
        else:
            client = boto3.client('sagemaker-runtime', region_name=self.input_param['region'])

        input_data = apply_formatting(preprocess_str(prompt))

        # Convert input data to JSON string
        payload = json.dumps(input_data)

        # Invoke the endpoint
        response = client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=payload,
            ContentType='application/json'
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())[0]["generated_text"]

        return idx, result


def call_diarization(prompts, endpoint_name, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    This function calls the speaker diarization model.

    Parameters:
    ...........
    prompts: dict
        Dictionary of diarized text chunks.
        key: chunk index, value: diarized text chunk.
    endpoint_name: str
        Name of the SageMaker endpoint.
    input_param: dict
        Additional arguments for the API call.

    Returns:
    ...........
    results: dict
        Dictionary of model outputs.
        key: chunk index, value: model output.

    ------------------------------------------------------------------------------------------------------
    """

    processor = AWSProcessor(endpoint_name, input_param)
    results = {}
    if input_param['parallel_processing'] == 1:
        with Pool(processes=len(prompts)) as pool:
            args = [(idx, prompts[idx]) for idx in sorted(prompts.keys())]
            results = pool.map(processor.process_chunk, args)
            results = dict(results)
    else:
        for idx in sorted(prompts.keys()):
            results[idx] = processor.process_chunk((idx, prompts[idx]))[1]

    return results


def correct_transcription(transcript_json, prompts, results, translate_json, asr):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs completion parsing of the LLM diarization model outputs
     and reinserts the corrected speaker labels into the transcription JSON.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    prompts: dict
        Dictionary of diarized text chunks.
        key: chunk index, value: diarized text chunk.
    results: dict
        Dictionary of model outputs.
        key: chunk index, value: model output.
    translate_json: dict
        Dictionary to translate '1' and '2' to the original speaker labels.
    asr: str
        Automatic Speech Recognition (ASR)
         system used to transcribe the audio.

    Returns:
    ...........
    dict: JSON response object with corrected speaker diarization.

    ------------------------------------------------------------------------------------------------------
    """
    speakers_pred = extract_predicted_speakers(prompts, results, translate_json)

    if asr == "aws":
        transcript_json_corrected = modify_transcription_aws(
            transcript_json, speakers_pred
        )
    elif asr == "whisperx":
        transcript_json_corrected = modify_transcription_whisperx(
            transcript_json, speakers_pred
        )
    else:
        raise ValueError("ASR not supported")

    return transcript_json_corrected

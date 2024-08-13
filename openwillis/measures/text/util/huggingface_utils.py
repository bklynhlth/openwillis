# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from openwillis.measures.text.util.diarization_utils import (
    preprocess_str, apply_formatting
)


def process_chunk_hf(args):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes a diarized text chunk using the HuggingFace model;
     it is used for parallel processing.

    Parameters:
    ...........
    args: tuple
        Tuple of arguments.
        idx: int, chunk index.
        prompt: str, diarized text chunk.
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.

    Returns:
    ...........
    tuple: Tuple of chunk index and model output.

    ------------------------------------------------------------------------------------------------------
    """
    idx, prompt, model, tokenizer = args

    input_data = apply_formatting(preprocess_str(prompt))

    # Tokenize the input data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(input_data["inputs"], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the output
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        stop_strings=["</s>", "###"],
        tokenizer=tokenizer,
        return_dict_in_generate=True
    )

    # Decode the output
    result = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

    return idx, result


def call_diarization_hf(prompts, model_name, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    This function calls the speaker diarization model.

    Parameters:
    ...........
    prompts: dict
        Dictionary of diarized text chunks.
        key: chunk index, value: diarized text chunk.
    model_name: str
        The name of the model used for transcription in HuggingFace.
    input_param: dict
        Additional arguments for the API call.

    Returns:
    ...........
    results: dict
        Dictionary of model outputs.
        key: chunk index, value: model output.

    ------------------------------------------------------------------------------------------------------
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=input_param['huggingface_token'])
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, token=input_param['huggingface_token'], device_map="cuda:0")
    else:
        raise ValueError("CUDA is not available.")

    results = {}
    for idx in sorted(prompts.keys()):
        results[idx] = process_chunk_hf((idx, prompts[idx], model, tokenizer))[1]

    return results

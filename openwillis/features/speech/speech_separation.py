# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
from diart.inference import Benchmark
from diart.models import SegmentationModel, EmbeddingModel
from diart import PipelineConfig, OnlineSpeakerDiarization
from pyannote.audio import Pipeline
from openwillis.features.speech import util as ut

import os
import shutil
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def run_diard(file_path, temp_dir, temp_rttm, hf_token):
    try:
    
        segmentation = SegmentationModel.from_pyannote("pyannote/segmentation", use_hf_token=hf_token)
        embedding = EmbeddingModel.from_pyannote("pyannote/embedding", use_hf_token=hf_token)
        config = PipelineConfig(segmentation=segmentation, embedding=embedding)

        with open(os.path.join(temp_rttm, file_path + '.rttm'), 'w') as fp:
            fp.write("SPEAKER 2_mix 1 0.230 0.717 <NA> <NA> speaker0 <NA> <NA>") #Standard txt for temp rttm file
            pass

        pipeline = OnlineSpeakerDiarization(config)
        benchmark = Benchmark(temp_dir, temp_rttm, temp_dir)
        benchmark(pipeline)
        
    except Exception as e:
        logger.info('Error in diard processing')
        
def run_pyannote(file_path, hf_token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    
    diart = pipeline(file_path, num_speakers=2)
    diart_df = ut.get_diart_interval(diart)
    diart_df = diart_df.sort_values(by=['start_time', 'end_time']).reset_index(drop=True)
    return diart_df

def process_diart(out_dir, file_name, filepath, hf_token):
    temp_dir, temp_rttm = ut.temp_process(out_dir, file_name, filepath)
    
    run_diard(file_name, temp_dir, temp_rttm, hf_token)
    rttm_df = ut.read_rttm(temp_dir, file_name)
    rttm_df = rttm_df.sort_values(by=['start_time', 'end_time']).reset_index(drop=True)
    
    ut.clean_prexisting(temp_dir, temp_rttm)
    return rttm_df

def speaker_separation(filepath, out_dir, hf_token, model='pyannote'):
    """
    -----------------------------------------------------------------------------------------
    
    Speech separation using Speech diarization 
    
    Args:
        filepath: audio file path 
        out_dir: Output directory
        
    Returns:
        results: Speaker separation
        
    -----------------------------------------------------------------------------------------
    """
    file_name, _ = os.path.splitext(os.path.basename(filepath))
    
    try:
        if os.path.exists(filepath):

            if model == 'pyannote-diart':
                rttm_df = process_diart(out_dir, file_name, filepath, hf_token)

            else:
                rttm_df = run_pyannote(filepath, hf_token)

            ut.slice_audio(rttm_df, filepath, out_dir)
            return rttm_df
        
    except Exception as e:
        logger.info('Error in diard processing....')


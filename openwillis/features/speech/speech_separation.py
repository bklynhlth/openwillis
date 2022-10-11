# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
from diart.inference import Benchmark
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig
from diart.models import SegmentationModel

import os
from openwillis.features.speech import util as ut
import shutil

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def run_diard(file_path, temp_dir, temp_rttm):
    try:
        
        config = PipelineConfig(
            segmentation=SegmentationModel.from_pyannote("pyannote/segmentation@Interspeech2021"))

        with open(os.path.join(temp_rttm, file_path + '.rttm'), 'w') as fp:
            fp.write("SPEAKER 2_mix 1 0.230 0.717 <NA> <NA> speaker0 <NA> <NA>") #Standard txt for temp rttm file
            pass

        pipeline = OnlineSpeakerDiarization(config)
        benchmark = Benchmark(temp_dir, temp_rttm, temp_dir)
        benchmark(pipeline)
        
    except Exception as e:
        logger.info('Error in diard processing')

def speaker_separation(filepath, out_dir):
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
            temp_dir, temp_rttm = ut.temp_process(out_dir, file_name, filepath)

            run_diard(file_name, temp_dir, temp_rttm)
            rttm_df = ut.read_rttm(temp_dir, file_name)

            ut.clean_prexisting(temp_dir, temp_rttm)
            ut.slice_audio(rttm_df, filepath, out_dir)
            return rttm_df
    except Exception as e:
        logger.info('Error in diard processing')


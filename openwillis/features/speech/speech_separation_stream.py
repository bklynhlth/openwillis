# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import os
from openwillis.features.speech import util as ut
import shutil

import subprocess
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def run_diard(file_path):
    try:
        
        output = subprocess.run(["diart.stream", file_path], capture_output=True)
        
    except Exception as e:
        logger.info('Error in diard processing')

def speaker_separation_stream(filepath, out_dir):
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
    dir_name = os.path.dirname(filepath)
    rttm_df = pd.DataFrame()
    
    try:
        if os.path.exists(filepath):
            ut.make_dir(out_dir)
            
            run_diard(filepath)
            rttm_df = ut.read_rttm(dir_name, file_name)
            
            rttm_df = rttm_df.sort_values(by=['start_time']).reset_index(drop=True)
            rttm_filepath = os.path.join(dir_name,file_name+'.rttm')

            if os.path.isfile(rttm_filepath):
                os.remove(rttm_filepath)

            ut.slice_audio(rttm_df, filepath, out_dir)
    except Exception as e:
        logger.info('Error in diard processing')
    
    return rttm_df

